import argparse
import time
import random
from itertools import chain
from types import SimpleNamespace
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist
from copy import deepcopy

from torch.profiler import profile, record_function, ProfilerActivity

import medusa

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def print_ids(tokenizer: AutoTokenizer, ids: torch.Tensor, prefix = "message:") -> None:
    text = tokenizer.decode(ids[0], skip_special_tokens=False)
    print(f"{prefix}: {text}")

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
    tokenizer: AutoTokenizer = None,
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

    if block_size > 1:
        medusa_buffers = medusa.generate_medusa_buffers(medusa.dflash_choice, device=model.device)
        medusa_attn_mask = medusa_buffers["medusa_attn_mask"].unsqueeze(0).unsqueeze(0).repeat(1, 32, 1, 1)
    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    topk = 15
    topk_iters = 1

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
            draft_ids = sample(draft_logits, topk=topk)
            block_output_ids[:, 1:] = draft_ids[:,:,0]
            # print('=' * 20)
            # for i in range(draft_ids.size(2)):
            #     print_ids(tokenizer, torch.cat([block_output_ids[:, :1], draft_ids[:,:,i]], dim=1), prefix=f"[yellow]Draft top{i+1} at position {start}[/yellow]")
            # print('=' * 20)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

            # past_key_values_target_snapshot = deepcopy(past_key_values_target)
            # output = target(
            #     block_output_ids,
            #     position_ids=block_position_ids,
            #     past_key_values=past_key_values_target_snapshot,
            #     use_cache=True,
            #     output_hidden_states=True if block_size > 1 else False,
            # )

            # posterior = sample(output.logits, temperature)
            # # print_ids(tokenizer, torch.cat([block_output_ids[:, :1], posterior[:,1:]], dim=1), prefix=f"[green]Posterior at position {start}[/green]")
            # seq_len = block_output_ids.shape[1] - 1
            # acceptance_length = 0
            # for i in range(seq_len):
            #     if torch.isin(posterior[:, i], draft_ids[:, i]):
            #         block_output_ids[:, i+1] = posterior[:, i]
            #         # # 打印posterior[:, i]是在draft_ids[:, i]的第几个
            #         # index_in_draft = (draft_ids[:, i] == posterior[:, i])[0].nonzero(as_tuple=True)[0].item()
            #         # print(f"[blue]Accepted token at position {i}, index in draft: {index_in_draft}[/blue]")
            #     else:
            #         break

            cart_candidates, tree_candidates = medusa.generate_candidates(
                draft_ids=draft_ids,
                verified_ids=block_output_ids[:, :1],
                tree_indices=medusa_buffers["tree_indices"],
                retrieve_indices=medusa_buffers["retrieve_indices"],
            )
            print("cart_candidates shape:", cart_candidates.shape)
            print("tree_candidates shape:", tree_candidates.shape)
            print("medusa_attn_mask shape:",medusa_attn_mask.shape)
            block_position_ids = (medusa_buffers["medusa_position_ids"] + start).unsqueeze(0)
            block_output_ids = tree_candidates
            attention_mask = torch.cat([torch.ones((1, 32, medusa_attn_mask.shape[2], start), device=medusa_attn_mask.device), medusa_attn_mask], dim=3)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                attention_mask=attention_mask,
                output_hidden_states=True if block_size > 1 else False,
            )
            target_logits = output.logits[0, medusa_buffers["retrieve_indices"]]
            posterior = sample(target_logits, temperature)
            print("posterior shape:", posterior.shape)
            acceptance_mask = (cart_candidates[0, :, 1:] == posterior[:, :-1]).int()
            candidates_accept_length = (torch.cumprod(acceptance_mask, dim=1)).sum(dim=1)
            print("candidates_accept_length:", candidates_accept_length)
            accept_length = candidates_accept_length.max()
            if accept_length > 0:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
                block_output_ids[:, : accept_length + 1] = cart_candidates[:, best_candidate, : accept_length + 1]
                select_indices = (
                    medusa_buffers["retrieve_indices"][best_candidate, : accept_length + 1] + start
                )
                output_ids[:, start : start + accept_length + 1] = block_output_ids[:, : accept_length + 1]
                output_ids[:, start + accept_length + 1] = posterior[best_candidate, accept_length]
                acceptance_lengths.append(accept_length + 1)
                start += accept_length + 1
                past_key_values_target.batch_select_indices(select_indices)
                print(output.hidden_states.shape)

        else:
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            # if acceptance_length == block_size - 1:
            #     print_ids(tokenizer, block_output_ids, prefix=f"[blue]Fully accepted block at position {start}[/blue]")

            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

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
        attn_implementation="eager",
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
            for block_size in [args.block_size]:
                response[block_size] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    tokenizer=tokenizer,
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

    # for each_res in responses:
    #     print("=================================")
    #     # decode output
    #     print("-------- original output ---------")
    #     original_ids = each_res[1].output_ids[0, each_res[1].num_input_tokens:]
    #     original_text = tokenizer.decode(original_ids, skip_special_tokens=False)
    #     print(original_text)

    #     print("-------- dflash output ---------")
    #     dflash_ids = each_res[args.block_size].output_ids[0, each_res[args.block_size].num_input_tokens:]
    #     dflash_text = tokenizer.decode(dflash_ids, skip_special_tokens=False)
    #     print(dflash_text)

    print(f"Decoding speedup: {t1 / tb:.2f}")
    print("Oringianl output throughput: "
          f"{1 / t1:.2f} tokens/sec, "
          f"D-Flash output throughput: {1 / tb:.2f} tokens/sec")
    tau = np.mean([np.mean(r[args.block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[args.block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(args.block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

if __name__ == "__main__":
    main()