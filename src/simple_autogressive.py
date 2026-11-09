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




if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B", dtype=torch.float16, local_files_only=True).cuda()
    model.eval()

    n_messages = [
        [{"role": "user", "content": "你好吗"}],
        [{"role": "user", "content": "1+1等于几？"}],
        [{"role": "user", "content": "用一句话介绍北京。"}],
    ]
    for messages in n_messages:
        # messages = [
        #     {"role": "user", "content": "你好吗"}
        # ]
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

        # HF 官方实现
        reference_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
        )

        # 个人实现
        generate_ids = generate_no_cache(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                16,
                tokenizer.eos_token_id
            )

        print(f"reference_ids shape : {reference_ids.shape}")
        print(f"generate_ids shape : {generate_ids.shape}")

        assert torch.equal(reference_ids, generate_ids), "手写生成与 HF 的 token IDs 不一致"


