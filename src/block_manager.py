class BlockAllocator:
    def __init__(self, num_blocks: int):
        """num_blocks 必须大于 0，否则抛出 ValueError。"""
        if num_blocks <= 0: raise ValueError("num_blocks need > 0")
        self.free_blocks = [i for i in range(num_blocks)]
        self.allocated_blocks = set()
        self.num_blocks = num_blocks

    @property
    def num_free_blocks(self) -> int:
        """返回当前空闲块数量。"""
        return len(self.free_blocks)

    def allocate(self) -> int:
        """分配一个空闲块；耗尽时抛出 RuntimeError。"""
        if self.num_free_blocks <= 0: raise RuntimeError("No Free Blocks")
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int) -> None:
        """释放块；ID 越界或该块未被分配时，抛出 ValueError。"""
        if block_id >= self.num_blocks or block_id < 0: raise ValueError("block id 越界")
        if block_id not in self.allocated_blocks: raise ValueError("block id 未被分配，无法释放")
        self.allocated_blocks.remove(block_id)
        self.free_blocks.append(block_id)


class RequestBlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        """block_size 必须大于 0。内部持有一个 BlockAllocator。"""
        if block_size <= 0: raise ValueError("block_size 必须大于 0")
        self.num_blocks = num_blocks
        self.blockallocator = BlockAllocator(num_blocks)
        self.block_size = block_size
        self.request_id_dict: dict[str, list[int]] = {}
        self.request_num_tokens: dict[str, int] = {}


    def allocate_request(
        self, request_id: str, num_tokens: int
    ) -> list[int]:
        """
        为请求分配块，保存块表，并返回块表的副本。

        num_tokens 必须大于 0。
        请求 ID 已存在时抛出 ValueError。
        空闲块不足时抛出 RuntimeError，且不改变已有状态。
        """
        if num_tokens <= 0: raise ValueError("num_tokens 必须大于 0")
        if request_id in self.request_id_dict.keys(): raise ValueError("请求 ID 已存在")
        needed_size = (num_tokens + self.block_size - 1) // self.block_size
        if needed_size > self.blockallocator.num_free_blocks: raise RuntimeError("空闲块不足")
        allocate_ids = [self.blockallocator.allocate() for i in range(needed_size)]
        self.request_id_dict[request_id] = allocate_ids
        self.request_num_tokens[request_id] = num_tokens
        return allocate_ids.copy()

    def free_request(self, request_id: str) -> None:
        """
        归还该请求的全部物理块，并删除请求映射。
        请求不存在时抛出 ValueError，且不改变状态。
        """
        if request_id not in self.request_id_dict.keys(): raise ValueError("请求不存在")
        free_list = self.request_id_dict[request_id]
        for i in free_list:
            self.blockallocator.free(i)

        _ = self.request_id_dict.pop(request_id)
        _ = self.request_num_tokens.pop(request_id)

    def append_token(self, request_id: str) -> None:
        """
        为请求追加一个 token 的容量。

        请求不存在：抛出 ValueError。
        需要新块但空闲块不足：抛出 RuntimeError，所有状态不变。
        成功：必要时追加一个物理块，并将 token 数增加 1。
        """
        if request_id not in self.request_id_dict.keys(): raise ValueError("请求不存在")
        now_num_tokens = self.request_num_tokens[request_id]
        if now_num_tokens % self.block_size == 0: # 此时已满，需要分配新的block
            new_allocate_id = self.blockallocator.allocate()
            self.request_id_dict[request_id].append(new_allocate_id)
        self.request_num_tokens[request_id] += 1

    def locate_token(
        self, request_id: str, token_index: int
    ) -> tuple[int, int]:
        """
        token_index 从 0 开始。
        返回 (物理块 ID, 块内偏移)。

        请求不存在，或 token_index 不在当前请求长度范围内：
            抛出 ValueError。
        查询不得修改状态。
        """
        if request_id not in self.request_id_dict.keys(): raise ValueError("请求不存在")
        if token_index < 0 or token_index >= self.request_num_tokens[request_id]: raise ValueError("token_index 不在当前请求长度范围内")
        block_id_index = token_index // self.block_size
        block_offset = token_index % self.block_size
        return (self.request_id_dict[request_id][block_id_index], block_offset)

def test_allocate_request():
    manager = RequestBlockManager(5, 4)

    a_blocks = manager.allocate_request("A", 5)
    assert len(a_blocks) == 2
    assert_consistent(manager)

    b_blocks = manager.allocate_request("B", 8)
    assert len(b_blocks) == 2
    assert set(a_blocks).isdisjoint(b_blocks)
    assert manager.blockallocator.num_free_blocks == 1
    assert_consistent(manager)

    blocks_before = {
        key: value.copy()
        for key, value in manager.request_id_dict.items()
    }
    tokens_before = manager.request_num_tokens.copy()

    try:
        manager.allocate_request("C", 5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("容量不足时应抛出 RuntimeError")

    assert manager.blockallocator.num_free_blocks == 1
    assert manager.request_id_dict == blocks_before
    assert manager.request_num_tokens == tokens_before
    assert_consistent(manager)

    c_blocks = manager.allocate_request("C", 1)
    assert len(c_blocks) == 1
    assert manager.blockallocator.num_free_blocks == 0
    assert_consistent(manager)

    try:
        manager.allocate_request("A", 5)
    except ValueError:
        pass
    else:
        raise AssertionError("重复请求应抛出 ValueError")
    assert_consistent(manager)

    a_blocks.append(999)
    assert manager.request_id_dict["A"] == blocks_before["A"]
    assert_consistent(manager)


def test_free_request():
    manager = RequestBlockManager(5, 4)

    manager.allocate_request("A", 5)
    assert_consistent(manager)

    b_blocks = manager.allocate_request("B", 8)
    assert manager.blockallocator.num_free_blocks == 1
    assert_consistent(manager)

    manager.free_request("A")
    assert manager.blockallocator.num_free_blocks == 3
    assert "A" not in manager.request_id_dict
    assert "A" not in manager.request_num_tokens
    assert manager.request_id_dict["B"] == b_blocks
    assert_consistent(manager)

    c_blocks = manager.allocate_request("C", 12)
    assert len(c_blocks) == 3
    assert set(c_blocks).isdisjoint(b_blocks)
    assert manager.blockallocator.num_free_blocks == 0
    assert_consistent(manager)

    try:
        manager.free_request("A")
    except ValueError:
        pass
    else:
        raise AssertionError("重复释放请求应抛出 ValueError")

    assert manager.blockallocator.num_free_blocks == 0
    assert_consistent(manager)

    manager.free_request("B")
    assert_consistent(manager)

    manager.free_request("C")
    assert manager.blockallocator.num_free_blocks == 5
    assert manager.blockallocator.allocated_blocks == set()
    assert manager.request_id_dict == {}
    assert manager.request_num_tokens == {}
    assert_consistent(manager)


def test_append_token():
    manager = RequestBlockManager(3, 4)

    a_blocks = manager.allocate_request("A", 3)
    assert len(a_blocks) == 1
    assert_consistent(manager)

    manager.append_token("A")
    assert manager.request_num_tokens["A"] == 4
    assert manager.request_id_dict["A"] == a_blocks
    assert manager.blockallocator.num_free_blocks == 2
    assert_consistent(manager)

    manager.append_token("A")
    assert manager.request_num_tokens["A"] == 5
    assert len(manager.request_id_dict["A"]) == 2
    assert manager.request_id_dict["A"][:1] == a_blocks
    assert manager.blockallocator.num_free_blocks == 1
    assert_consistent(manager)

    manager.allocate_request("B", 4)
    assert manager.blockallocator.num_free_blocks == 0
    assert_consistent(manager)

    for expected_length in (6, 7, 8):
        manager.append_token("A")
        assert manager.request_num_tokens["A"] == expected_length
        assert len(manager.request_id_dict["A"]) == 2
        assert_consistent(manager)

    blocks_before = manager.request_id_dict["A"].copy()
    free_before = manager.blockallocator.free_blocks.copy()
    allocated_before = manager.blockallocator.allocated_blocks.copy()

    try:
        manager.append_token("A")
    except RuntimeError:
        pass
    else:
        raise AssertionError("容量不足时应抛出 RuntimeError")

    assert manager.request_num_tokens["A"] == 8
    assert manager.request_id_dict["A"] == blocks_before
    assert manager.blockallocator.free_blocks == free_before
    assert manager.blockallocator.allocated_blocks == allocated_before
    assert_consistent(manager)

    manager.free_request("B")
    assert manager.blockallocator.num_free_blocks == 1
    assert_consistent(manager)

    manager.append_token("A")
    assert manager.request_num_tokens["A"] == 9
    assert len(manager.request_id_dict["A"]) == 3
    assert manager.request_id_dict["A"][:2] == blocks_before
    assert manager.blockallocator.num_free_blocks == 0
    assert_consistent(manager)

    manager.free_request("A")
    assert manager.blockallocator.num_free_blocks == 3
    assert manager.request_id_dict == {}
    assert manager.request_num_tokens == {}
    assert_consistent(manager)

    try:
        manager.append_token("missing")
    except ValueError:
        pass
    else:
        raise AssertionError("请求不存在时应抛出 ValueError")
    assert_consistent(manager)


def test_locate_token():
    manager = RequestBlockManager(5, 4)
    blocks = manager.allocate_request("A", 9)
    assert_consistent(manager)

    blocks_before = manager.request_id_dict["A"].copy()
    tokens_before = manager.request_num_tokens.copy()
    free_before = manager.blockallocator.free_blocks.copy()
    allocated_before = manager.blockallocator.allocated_blocks.copy()

    cases = [
        (0, (blocks[0], 0)),
        (3, (blocks[0], 3)),
        (4, (blocks[1], 0)),
        (7, (blocks[1], 3)),
        (8, (blocks[2], 0)),
    ]
    for token_index, expected in cases:
        assert manager.locate_token("A", token_index) == expected
        assert_consistent(manager)

    for token_index in (-1, 9):
        try:
            manager.locate_token("A", token_index)
        except ValueError:
            pass
        else:
            raise AssertionError("索引越界时应抛出 ValueError")
        assert_consistent(manager)

    try:
        manager.locate_token("missing", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("请求不存在时应抛出 ValueError")
    assert_consistent(manager)

    assert manager.request_id_dict["A"] == blocks_before
    assert manager.request_num_tokens == tokens_before
    assert manager.blockallocator.free_blocks == free_before
    assert manager.blockallocator.allocated_blocks == allocated_before

def assert_consistent(manager: RequestBlockManager) -> None:
    """检查块分配器与请求记录之间的一致性，失败时触发断言。"""
    assert manager.request_id_dict.keys() == manager.request_num_tokens.keys() # 两个请求字典拥有相同的请求 ID
    all_request_list = []
    for key in manager.request_id_dict.keys():
        assert len(manager.request_id_dict[key]) == (manager.request_num_tokens[key] + manager.block_size - 1) // manager.block_size # 每个请求的块数等于：(num_tokens + block_size - 1) // block_size
        all_request_list += manager.request_id_dict[key]
    assert len(set(all_request_list)) == len(all_request_list) # 汇总所有请求块表，确保没有重复物理块
    assert sorted(all_request_list) == sorted(manager.blockallocator.allocated_blocks) # 请求持有的全部块，恰好等于 allocated_blocks
    assert len(set(manager.blockallocator.free_blocks)) == len(manager.blockallocator.free_blocks) # 空闲列表没有重复 ID
    assert len(set(all_request_list + manager.blockallocator.free_blocks)) == len(all_request_list + manager.blockallocator.free_blocks) # 空闲列表与已分配集合不相交
    assert set(all_request_list + manager.blockallocator.free_blocks) == set(range(manager.blockallocator.num_blocks))

if __name__ == '__main__':
    test_allocate_request()
    
    test_free_request()

    test_append_token()

    test_locate_token()
