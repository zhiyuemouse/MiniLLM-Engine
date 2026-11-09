import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def select_next_token(logits: torch.Tensor) -> torch.Tensor:
    """从 [B, T, V] 的 logits 中选择下一 token，返回 [B, 1]。"""
    # TODO：检查输入是三维张量，且每个维度都大于 0
    # TODO：取最后一个时间位置
    # TODO：沿词表维度取最大值索引，保留维度
    if logits.ndim != 3:
        raise ValueError("logits must have 3 dimensions")
    if 0 in logits.shape:
        raise ValueError("dim 0 in logits shape")

    output = logits[:, -1, :] # [B, V]
    output = torch.argmax(output, dim=-1, keepdim=True) # [B, 1]
    return output

def generate_no_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    """
    限定：单请求，无 padding，贪心解码。
    返回：完整 token IDs，包含原始输入和新生成部分。
    max_new_tokens=0 时返回原始序列。
    生成 EOS 后保留该 token，并立即停止。
    """
    with torch.inference_mode():
        generate_ids = input_ids
        for step in range(max_new_tokens):
            output = model(
                input_ids=generate_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            next_token_ids = select_next_token(output.logits) # [B, 1]
            generate_ids = torch.cat([generate_ids, next_token_ids], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
            if next_token_ids.item() == eos_token_id:
                break

    return generate_ids

def generate_with_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    """
    单请求、无 padding、贪心解码。
    返回完整 token IDs，包含 prompt 和可能生成的 EOS。
    max_new_tokens=0 时直接返回原输入。
    """
    generate_ids = input_ids
    current_input_ids = generate_ids
    past_key_values = None
    with torch.inference_mode():
        for step in range(max_new_tokens):
            print(f"实际输入长度: {current_input_ids.shape[1]}")
            output = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = output.past_key_values
            print(f"前向后的缓存长度: {past_key_values.get_seq_length()}")
            current_input_ids = select_next_token(output.logits) # [B, 1]
            generate_ids = torch.cat([generate_ids, current_input_ids], dim=1)
            if (current_input_ids.item() == eos_token_id):
                break
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)

        return generate_ids


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B", dtype=torch.float16, local_files_only=True).cuda()
    model.eval()

    prompts = [
        "你好吗",
        "1+1等于几？",
        "用一句话介绍北京。",
    ]
    generation_lengths = [0, 1, 16]

    for i in range(len(prompts)):

        messages = [
            {"role": "user", "content": f"{prompts[i]}"}
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

        for j in range(len(generation_lengths)):
            generate_no_cache_output = generate_no_cache(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                generation_lengths[j],
                tokenizer.eos_token_id
            )

            generate_with_cache_output = generate_with_cache(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                generation_lengths[j],
                tokenizer.eos_token_id
            )

            assert torch.equal(generate_no_cache_output, generate_with_cache_output), "断言完整 token IDs 不一致"

            assert torch.equal(generate_with_cache_output[0, :prompt_len], inputs["input_ids"][0]), "断言返回序列的前缀与输入不一致"

            assert (generate_with_cache_output.shape[1] - prompt_len) <= generation_lengths[j], "断言新增 token 数超过上限"

            print(f"prompt: {prompts[i]}")
            print(f"生成上限: {generation_lengths[j]}")
            print(f"实际新增数: {generate_with_cache_output.shape[1] - prompt_len}")
            print("PASS")

    
