import torch
from block_manager import RequestBlockManager, assert_consistent


class BlockStorage:
    def __init__(self, manager: RequestBlockManager, head_dim: int):
        """
        保存 manager 引用。
        head_dim 必须大于 0，否则抛出 ValueError。
        创建 CPU float32 零张量：
            [num_blocks, block_size, head_dim]
        """
        if head_dim <= 0: raise ValueError("head_dim 必须大于 0")
        self.manager = manager
        self.head_dim = head_dim
        self.tensor = torch.zeros((manager.num_blocks, manager.block_size, head_dim), dtype=torch.float32, device="cpu")

    def write_token(
        self,
        request_id: str,
        token_index: int,
        value: torch.Tensor,
    ) -> None:
        """
        value 限定为 CPU float32，形状 [head_dim]。
        不符合要求时抛出 ValueError，存储不变。
        通过 locate_token 定位后写入。
        """
        if value.shape != (self.head_dim,):
            raise ValueError(f"value 的形状必须为 ({self.head_dim},)")
        if value.dtype != torch.float32:
            raise ValueError(f"value 的dtype必须为 torch.float32")
        if value.device.type != "cpu":
            raise ValueError("value 必须位于 CPU")

        (block_id, block_offset) = self.manager.locate_token(request_id, token_index)
        self.tensor[block_id, block_offset] = value

    def read_token(
        self, request_id: str, token_index: int
    ) -> torch.Tensor:
        """定位并返回该位置数据的副本，避免外部修改存储。"""
        (block_id, block_offset) = self.manager.locate_token(request_id, token_index)
        return self.tensor[block_id, block_offset].clone()

    def read_request(self, request_id: str) -> torch.Tensor:
        """
        返回请求全部有效 token 的数据，形状 [num_tokens, head_dim]。
        不包含末块尚未使用的位置。
        返回独立副本。
        请求不存在时抛出 ValueError。
        """
        if request_id not in self.manager.request_id_dict.keys(): raise ValueError("请求不存在")
        num_tokens = self.manager.request_num_tokens[request_id]
        ans = []
        for i in range(num_tokens):
            per_token = self.read_token(request_id, i)
            ans.append(per_token)
        ans = torch.stack(ans, dim=0)
        return ans

def test_read_request():
    manager = RequestBlockManager(5, 4)
    blockstorage = BlockStorage(manager, 2)

    # A 分配 5 个 token，B 分配 4 个 token。
    A_blocks = manager.allocate_request('A', 5)
    B_blocks = manager.allocate_request('B', 4)
    # 为 A 的位置 0～4 分别写入 [i, -i]。
    for i in range(5):
        blockstorage.write_token('A', i, torch.tensor([i, -i], dtype=torch.float32))
    # 给 A 连续追加 4 个 token，使长度达到 9，再写入位置 5～8
    manager.append_token('A')
    manager.append_token('A')
    manager.append_token('A')
    manager.append_token('A')
    for i in range(5, 9):
        blockstorage.write_token('A', i, torch.tensor([i, -i], dtype=torch.float32))

    # 调用 read_request("A")
    A_token = blockstorage.read_request('A')
    assert A_token.shape == (9, 2) # 形状为 [9, 2]
    for i in range(9):
        assert torch.equal(A_token[i], torch.tensor([i, -i], dtype=torch.float32)) # 数据逐行等于 [i, -i]，顺序不乱
    assert A_token.shape[0] == 9 # 不包含末块剩余的 3 个位置
    # 原地修改返回张量，再次读取，确认内部存储不变
    before = blockstorage.read_request('A')
    A_token[5] = torch.tensor([100, 200], dtype=torch.float32)
    after = blockstorage.read_request('A')
    assert torch.equal(before, after)

    # 不存在的请求抛出 ValueError
    try:
        nonexistent = blockstorage.read_request('missing')
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")

def test_block_storage():
    manager = RequestBlockManager(5, 4)
    blockstorage = BlockStorage(manager, 2)

    # A 分配 5 个 token，B 分配 4 个 token。
    A_blocks = manager.allocate_request('A', 5)
    B_blocks = manager.allocate_request('B', 4)

    # 为 A 的位置 0～4 分别写入 [i, -i]，逐个读回并断言相等，覆盖跨块边界。
    for i in range(5):
        blockstorage.write_token('A', i, torch.tensor([i, -i], dtype=torch.float32))
    for i in range(5):
        A_values = blockstorage.read_token('A', i)
        assert torch.equal(A_values, torch.tensor([i, -i], dtype=torch.float32))

    # B 的位置 0 写入 [100, 200]，确认 A 的数据不受影响。
    blockstorage.write_token('B', 0, torch.tensor([100, 200], dtype=torch.float32))
    for i in range(5):
        A_values = blockstorage.read_token('A', i)
        assert torch.equal(A_values, torch.tensor([i, -i], dtype=torch.float32))

    # 修改 read_token() 返回的张量，确认内部数据不变。
    before = blockstorage.read_token('A', i)
    storage_before = blockstorage.tensor.clone()
    A_values = blockstorage.read_token('A', i)
    A_values.fill_(99)
    assert torch.equal(before, blockstorage.read_token('A', i))
    assert torch.equal(storage_before, blockstorage.tensor.clone())

    # 尝试写入形状 [3] 或 dtype 为整数的张量：抛出 ValueError，底层存储不变。
    try:
        blockstorage.write_token('A', 0, torch.tensor([1, 1, 1], dtype=torch.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")
    assert torch.equal(storage_before, blockstorage.tensor)
    
    try:
        blockstorage.write_token('A', 0, torch.tensor([1, 1], dtype=torch.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")
    assert torch.equal(storage_before, blockstorage.tensor)

    # 查询不存在的请求或越界位置：沿用 locate_token() 的异常
    try:
        blockstorage.read_token('missing', 0)
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")

    try:
        blockstorage.read_token('A', 14)
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")

def test_block_reuse():
    # 创建总块数为 3、块大小为 4、`head_dim=2` 的管理器和存储。
    manager = RequestBlockManager(3, 4)
    blockstorage = BlockStorage(manager, 2)
    # A 申请 5 个 token，B 申请 4 个 token，耗尽全部块
    A_blocks = manager.allocate_request('A', 5)
    B_blocks = manager.allocate_request('B', 4)
    # 给 A、B 的所有有效位置写入不同的数据：
    for i in range(5):
        blockstorage.write_token('A', i, torch.tensor([i, -i], dtype=torch.float32, device="cpu"))
    for i in range(4):
        blockstorage.write_token('B', i, torch.tensor([100+i, 200+i], dtype=torch.float32, device="cpu"))
    # 保存 A 的块表副本，以及 B 的完整数据副本。
    A_blocks_list = manager.request_id_dict['A'].copy()
    for i in range(4):
        assert torch.equal(blockstorage.read_token('B', i), torch.tensor([100+i, 200+i], dtype=torch.float32, device="cpu"))
    B_blocks_val = blockstorage.read_request('B').clone()
    # 释放 A，再让 C 申请 5 个 token。
    manager.free_request('A')
    manager.allocate_request('C', 5)
    # 给 C 的全部有效位置写入 [300+i, 400+i]
    for i in range(5):
        blockstorage.write_token('C', i, torch.tensor([300+i, 400+i], dtype=torch.float32, device="cpu"))

    # C 使用的物理块集合等于 A 原来的块集合；不要求块表顺序相同。
    C_blocks_list = manager.request_id_dict['C'].copy()
    assert sorted(A_blocks_list) == sorted(C_blocks_list)

    # C 读回的数据全部是新写入值。
    for i in range(5):
        assert torch.equal(blockstorage.read_token('C', i), torch.tensor([300+i, 400+i], dtype=torch.float32, device="cpu"))

    # B 的数据保持不变
    assert torch.equal(B_blocks_val, blockstorage.read_request('B'))

    # 释放后的 A 无法再通过 read_request("A") 读取，抛出 ValueError。
    try:
        _ = blockstorage.read_request('A')
    except ValueError:
        pass
    else:
        raise AssertionError("未抛出对应错误")

    # 操作完成后，`assert_consistent(manager)` 通过
    assert_consistent(manager)

if __name__ == "__main__":
    test_block_storage()
    print("test_block_storage PASS")

    test_read_request()
    print("test_read_request PASS")

    test_block_reuse()
    print("test_block_reuse PASS")
