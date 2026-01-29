import argparse
import time
import random
from itertools import chain
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

from torch.profiler import profile, record_function, ProfilerActivity

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class NGramCache:
    """N-gram 缓存：用第一个 token 作为 key，记录后续 n-1 个 token"""
    
    def __init__(self, cache_size: int = 1000, n_gram: int = 3):
        self.cache_size = cache_size  # 缓存容量
        self.n_gram = n_gram  # 记录的总长度（1 个 key + n-1 个后续 token）
        # key: 单个 token, value: 后续 n-1 个 token 的 tuple
        self.cache = {}
        # FIFO 队列
        self.fifo_queue = []
    
    def update(self, token_sequence: torch.Tensor):
        """用验证通过的 token 序列更新缓存"""
        seq = token_sequence.squeeze().tolist()
        if isinstance(seq, int):
            seq = [seq]
        
        # 每个位置的 token 作为 key，记录后续 n-1 个 token
        for i in range(len(seq) - self.n_gram + 1):
            key_token = seq[i]
            following_tokens = tuple(seq[i + 1 : i + self.n_gram])
            
            if key_token in self.cache:
                self.cache[key_token] = following_tokens
            else:
                if len(self.cache) >= self.cache_size:
                    oldest_key = self.fifo_queue.pop(0)
                    del self.cache[oldest_key]
                
                self.cache[key_token] = following_tokens
                self.fifo_queue.append(key_token)
    
    def lookup(self, key_token: int) -> tuple | None:
        """查询 key_token 后续的 n-1 个 token"""
        if key_token in self.cache:
            return self.cache[key_token]
        return None

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    n_gram_cache: NGramCache | None = None,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    topk = 30

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            draft_ids = sample(draft_logits, topk=topk)  # [1, seq_len, topk]
            
            # 先用 top1 填充
            block_output_ids[:, 1:] = draft_ids[:, :, 0]
            
            # 用第一个 draft token (top1) 查表
            if n_gram_cache is not None:
                for idx, each_token in enumerate(block_output_ids[0, 1:].tolist()):
                    cached_following = n_gram_cache.lookup(each_token)
                    
                    if cached_following is not None:
                        # 检查缓存中记录的后续 token 是否在对应位置的 topk 中
                        for pos, cached_token in enumerate(cached_following):
                            if pos + idx + 1 >= block_size - 1:
                                break
                            topk_at_pos = draft_ids[0, pos + idx + 1, :].tolist()
                            if cached_token in topk_at_pos:
                                block_output_ids[0, pos + idx + 2] = cached_token
                            else:
                                # 后续不匹配了，停止替换
                                break
            
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

            # 只需要一次验证
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()

        else:
            # block_size == 1
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=False,
            )
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        # 用验证通过的序列更新 n-gram 缓存
        if n_gram_cache is not None and acceptance_length > 0:
            accepted_seq = output_ids[:, start : start + acceptance_length + 2]  # +2 包含 bonus token
            n_gram_cache.update(accepted_seq)

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--draft-name-or-path", type=str, default="None")
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache-size", type=int, default=10000, help="N-gram cache size (FIFO)")
    parser.add_argument("--n-gram", type=int, default=2, help="N-gram length for each cache entry")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    # 创建 n-gram 缓存，在所有样本间共享
    n_gram_cache = NGramCache(cache_size=args.cache_size, n_gram=args.n_gram)

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for block_size in [1, args.block_size]:
                response[block_size] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=[tokenizer.eos_token_id],
                    n_gram_cache=n_gram_cache if block_size > 1 else None,
                    temperature=args.temperature,
                )
            
            spec_response = response[args.block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[args.block_size].time_per_output_token for r in responses])

    for each_res in responses:
        print("=================================")
        # decode output
        print("-------- original output ---------")
        original_ids = each_res[1].output_ids[0, each_res[1].num_input_tokens:]
        original_text = tokenizer.decode(original_ids, skip_special_tokens=False)
        print(original_text)

        print("-------- dflash output ---------")
        dflash_ids = each_res[args.block_size].output_ids[0, each_res[args.block_size].num_input_tokens:]
        dflash_text = tokenizer.decode(dflash_ids, skip_special_tokens=False)
        print(dflash_text)

    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[args.block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[args.block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(args.block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

if __name__ == "__main__":
    main()