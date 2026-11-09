import torch
import torch.nn.functional as F
from block_manager import RequestBlockManager, assert_consistent
from block_storage import BlockStorage
import math

def single_query_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    CPU float32：
        query:  [D]
        keys:   [S, D]
        values: [S, D]
    返回：[D]
    本次假设形状合法，S、D 均大于 0。
    """
    Q = query.unsqueeze(0) # [1, D]
    attn_score = (Q @ keys.transpose(0, 1)) / math.sqrt(keys.shape[1]) # [1, S]
    attn_score = F.softmax(attn_score, dim=-1)
    output = attn_score @ values # [1, D]
    return output.squeeze(0) # [D]

def test_block_attention_matches_contiguous():
    # 设置固定随机种子，例如 torch.manual_seed(42)
    torch.manual_seed(42)
    # 使用 D=4、块大小 4、总块数 5
    manager = RequestBlockManager(5, 4)
    blockstorage_k = BlockStorage(manager, 4)
    blockstorage_v = BlockStorage(manager, 4)
    # A 先申请 5 个位置，B 申请 4 个位置，再给 A 追加至 9，形成不连续块表
    manager.allocate_request('A', 5)
    manager.allocate_request('B', 4)
    manager.append_token('A')
    manager.append_token('A')
    manager.append_token('A')
    manager.append_token('A')

    # 创建随机 query [4]、`keys [9, 4]`、`values [9, 4]`。
    query = torch.randn((4), dtype=torch.float32, device="cpu")
    keys = torch.randn((9, 4), dtype=torch.float32, device="cpu")
    values = torch.randn((9, 4), dtype=torch.float32, device="cpu")

    # 按 token 位置，把 K、V 分别写入两个存储。
    for i in range(0, 9):
        blockstorage_k.write_token('A', i, keys[i])
        blockstorage_v.write_token('A', i, values[i])

    # 用 read_request("A") 收集 K、V，调用你的 attention 函数
    k_cache = blockstorage_k.read_request('A')
    v_cache = blockstorage_v.read_request('A')
    attn_output_kvcache = single_query_attention(query, k_cache, v_cache)
    reference_scores = (keys @ query) / math.sqrt(query.shape[0])
    reference_output = torch.softmax(reference_scores, dim=0) @ values
    torch.testing.assert_close(k_cache, keys, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, values, rtol=0, atol=0)
    assert attn_output_kvcache.shape == (4,)
    torch.testing.assert_close(attn_output_kvcache, reference_output)
    assert_consistent(manager)

def test_attention_after_append():
    # 设置固定随机种子，例如 torch.manual_seed(42)
    torch.manual_seed(42)
    # 使用 D=4、块大小 4、总块数 5
    manager = RequestBlockManager(4, 4)
    blockstorage_k = BlockStorage(manager, 4)
    blockstorage_v = BlockStorage(manager, 4)
    # A 申请 8 个位置，B 申请 4 个位置。
    manager.allocate_request('A', 8)
    manager.allocate_request('B', 4)
    # 创建并写入 A 的历史 keys [8, 4]、`values [8, 4]`。
    history_k = torch.randn((8, 4), dtype=torch.float32, device="cpu")
    history_v = torch.randn((8, 4), dtype=torch.float32, device="cpu")
    for i in range(8):
        blockstorage_k.write_token('A', i, history_k[i])
        blockstorage_v.write_token('A', i, history_v[i])
    assert manager.request_num_tokens['A'] == 8
    assert len(manager.request_id_dict['A']) == 2
    # 创建新位置的 query [4]、`new_key [4]`、`new_value [4]`。
    query = torch.randn((4), dtype=torch.float32, device="cpu")
    new_key = torch.randn((4), dtype=torch.float32, device="cpu")
    new_value = torch.randn((4), dtype=torch.float32, device="cpu")
    # 保存 A 的旧块表副本，然后调用 append_token("A")。
    A_old_block_list = manager.request_id_dict['A'].copy()
    manager.append_token('A')
    # 将新 K、V 写入 A 的位置 8。
    blockstorage_k.write_token('A', 8, new_key)
    blockstorage_v.write_token('A', 8, new_value)
    assert manager.request_num_tokens['A'] == 9
    assert len(manager.request_id_dict['A']) == 3
    # 从分块存储读回全部 9 个位置，计算 attention。
    k_cache = blockstorage_k.read_request('A')
    v_cache = blockstorage_v.read_request('A')
    attn_output_kvcache = single_query_attention(query, k_cache, v_cache)
    # 用 torch.cat 将历史 K、V 与新向量拼成连续参考，再直接写公式计算参考输出。
    k_continues = torch.cat([history_k, new_key.unsqueeze(0)], dim=0)
    v_continues = torch.cat([history_v, new_value.unsqueeze(0)], dim=0)
    reference_scores = (k_continues @ query) / math.sqrt(query.shape[0])
    reference_output = torch.softmax(reference_scores, dim=0) @ v_continues
    assert manager.request_id_dict["A"][:2] == A_old_block_list
    assert manager.locate_token("A", 8) == (
        manager.request_id_dict["A"][2], 0
    )

    torch.testing.assert_close(k_cache, k_continues, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_continues, rtol=0, atol=0)
    torch.testing.assert_close(attn_output_kvcache, reference_output)
    assert_consistent(manager)


if __name__ == '__main__':
    test_block_attention_matches_contiguous()
    print("test_block_attention_matches_contiguous PASS")

    test_attention_after_append()
    print("test_attention_after_append PASS")