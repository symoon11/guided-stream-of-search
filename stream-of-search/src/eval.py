import os

os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/home/seungyong/huggingface"

import argparse
import json
import random
from typing import Dict, List

import numpy as np
import torch
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from countdown_utils import metric_fn

DATA_DIR = "/home/seungyong/train-countdown/stream-of-search/data/b4-rand-final"
DATA_FILES = {
    "train": "train1_b4_t100_n500000_random.json",
    "val": "val1_b4_t100_n500000_random.json",
    "test": "val_target1_b4_t100_n500000_random.json",
}


def get_prompts(samples: List[Dict[str, str]]) -> List[str]:
    """
    Returns prompts from the given samples.
    """
    prompts = []
    for sample in samples:
        optimal_path = sample["optimal_path"].strip()
        prompt = optimal_path.split("\n")[0] + "\n"
        prompts.append(prompt)
    return prompts


def get_search_paths(samples: List[Dict[str, str]]) -> List[str]:
    """
    Returns search paths from the given samples.
    """
    search_paths = []
    for sample in samples:
        search_path = sample["search_path"].strip()
        search_paths.append(search_path)
    return search_paths


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    prompt_len: int = 17,
    max_gen_len: int = 4079,
    temperature: float = 0.0,
    stop_strings: List[str] = ["Goal Reached", "Exited"],
) -> List[str]:
    """
    Generates search paths starting from the given prompts.
    """
    # Encode prompts
    inputs = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=prompt_len,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate tokens
    if temperature == 0.0:
        all_tokens = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            do_sample=False,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    else:
        all_tokens = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            do_sample=True,
            temperature=temperature,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Get search paths
    search_paths = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)

    return search_paths


def main(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build model
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model = model.eval()

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    # Load data
    data_file = os.path.join(DATA_DIR, DATA_FILES[args.split])
    with open(data_file, "r") as json_file:
        data = json.load(json_file)

    # Evaluate model
    start = args.start
    end = min(start + args.offset, len(data))
    trajectories = []
    gen_ratings = []
    ref_ratings = []

    for batch_start in trange(start, end, args.batch_size):
        # Get samples
        batch_end = min(batch_start + args.batch_size, end)
        samples = data[batch_start:batch_end]

        # Get prompts
        prompts = get_prompts(samples)

        # Get search paths
        ref_search_paths = get_search_paths(samples)

        # Generate search paths
        gen_search_paths = generate(
            model,
            tokenizer,
            prompts,
            prompt_len=args.prompt_len,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
        )

        # Update trajectories
        trajectories += gen_search_paths

        # Compute ratings
        for gen_search_path, ref_search_path in zip(gen_search_paths, ref_search_paths):
            gen_rating, _ = metric_fn(gen_search_path)
            ref_rating, _ = metric_fn(ref_search_path)
            gen_ratings.append(gen_rating)
            ref_ratings.append(ref_rating)

        # Compute stats
        gen_rating = np.mean(gen_ratings)
        ref_rating = np.mean(ref_ratings)
        gen_accuracy = np.mean([r > 0 for r in gen_ratings])
        ref_accuracy = np.mean([r > 0 for r in ref_ratings])

        # Print stats
        print()
        print(f"Gen Rating: {gen_rating}, Gen Accuracy: {gen_accuracy}")
        print(f"Ref Rating: {ref_rating}, Ref Accuracy: {ref_accuracy}")

    # Save stats
    stats = {
        "trajectories": trajectories,
        "gen_ratings": gen_ratings,
        "ref_ratings": ref_ratings,
    }
    stat_file = f"stats_final_{args.split}_{start}_{end}.json"
    stat_file = os.path.join(args.ckpt, stat_file)
    with open(stat_file, "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--offset", default=1000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--prompt_len", default=17, type=int)
    parser.add_argument("--max_gen_len", default=4079, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    args = parser.parse_args()

    # Run main
    main(args)
