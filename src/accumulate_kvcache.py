import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def estimate_kv_bytes(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    element_size: int,
) -> int:
    """根据模型结构计算理论 KV 张量字节数。"""
    # KV字节数 = 2 * L * B * S * H_n * H_d * E
    output = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * element_size
    return output


def count_kv_tensor_bytes(past_key_values) -> int:
    """遍历各层 K、V，累加 numel() * element_size()。"""
    output = 0
    print_kElementSize_flag = True
    for k, v in past_key_values:
        output += k.numel() * k.element_size()
        output += v.numel() * v.element_size()
        if print_kElementSize_flag:
            print(f"K 张量的 element_size : {k.element_size()}")
            print_kElementSize_flag = False

    return output


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B", dtype=torch.float16, local_files_only=True).cuda()
    model.eval()

    messages = [
        {"role": "user", "content": "用一句话介绍北京。"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=True
        )

    config = model.config
    print(f"num_hidden_layers : {config.num_hidden_layers}")
    print(f"num_key_value_heads : {config.num_key_value_heads}")
    print(f"head_dim : {config.head_dim}")
    print(f"实际缓存长度 : {output.past_key_values.get_seq_length()}")
    estimate_Bytes = estimate_kv_bytes(
        config.num_hidden_layers,
        1,
        inputs['input_ids'].shape[1],
        config.num_key_value_heads,
        config.head_dim,
        output.past_key_values[0][0].element_size()
    )
    count_Bytes = count_kv_tensor_bytes(
        output.past_key_values
    )
    estimate_1Token_Bytes = estimate_kv_bytes(
        config.num_hidden_layers,
        1,
        1,
        config.num_key_value_heads,
        config.head_dim,
        2
    )
    print(f"每 token 的理论字节数 : {estimate_1Token_Bytes}")
    print(f"estimate_kv_bytes : {estimate_Bytes / (1024**2)} MiB")
    print(f"count_kv_tensor_bytes : {count_Bytes / (1024**2)} MiB")


    assert estimate_Bytes == count_Bytes, "ERROR"