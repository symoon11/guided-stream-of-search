import os

os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/home/seungyong/huggingface"

import argparse
import json
import random
import re
from typing import Any, Dict, List, Tuple

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

DATA_DIR = "/home/seungyong/guided-stream-of-search/stream-of-search/data/b4-rand"
DATA_FILES = {
    "train": "train1_b4_t100_n500000_random.json",
    "val": "val1_b4_t100_n500000_random.json",
    "test": "val_target1_b4_t100_n500000_random.json",
}


def parse_operation(operation: str) -> Tuple[int, int, int]:
    """
    Parses the given operation and returns the operands and result.
    """

    # Split operation into lhs and rhs
    lhs, rhs = operation.split("=")

    # Match operands and result
    match = re.match(r"(\d+)[-+*/](\d+)", lhs)

    # Convert operands and result into integers
    operand1 = int(match.group(1))
    operand2 = int(match.group(2))
    result = int(rhs)

    return operand1, operand2, result


def get_subgoals_list(optimal_path: str) -> List[List[List[str]]]:
    """
    Returns a list of subgoals that need to be achieved to reach the goal.
    """

    # Match operations
    expl_pattern = r"Current State: \d+:\[.*?\], Operations: (\[.*?\])\n"
    expl_match = list(re.finditer(expl_pattern, optimal_path))[-1]
    ops = eval(expl_match.group(1))
    assert len(ops) == 2
    op1 = ops[0]
    op2 = ops[1]

    # Parse operations
    *op1_operands, op1_result = parse_operation(op1)
    *op2_operands, op2_result = parse_operation(op2)

    # Get subgoals list
    subgoals_list = []
    if op1_result in op2_operands:
        # Subgoals are dependent
        subgoals_list.append([[], [op1], [op1, op2]])
    else:
        # Subgoals are independent
        subgoals_list.append([[], [op1], [op1, op2]])
        subgoals_list.append([[], [op2], [op2, op1]])

    return subgoals_list


def get_subgoal_nodes(
    trajectory: str, subgoals_list: List[List[List[str]]], depth: int = 1
) -> Dict[str, Any]:
    """
    Returns a list of nodes that achieve the subgoal at the specified depth in the trajectory.
    """

    # Match explored nodes
    expl_pattern = (
        r"Moving to Node #(\d+(?:,\d+)*)\n"
        r"Current State: (\d+):(\[.*?\]), Operations: (\[.*?\])\n"
    )
    expl_matches = list(re.finditer(expl_pattern, trajectory))

    # Get subgoal nodes
    subgoal_nodes = {}
    for expl_match in expl_matches:
        # Get node information
        id = expl_match.group(1)
        target = int(expl_match.group(2))
        try:
            nums = eval(expl_match.group(3))
        except:
            continue
        try:
            ops = eval(expl_match.group(4))
        except:
            continue
        expl_start = expl_match.start()
        expl_end = expl_match.end()

        # Check if node achieves the subgoal at the specified depth
        if len(id.split(",")) == depth + 1 and len(ops) == depth:
            for subgoals in subgoals_list:
                if ops == subgoals[depth] and id not in subgoal_nodes:
                    node = {
                        "target": target,
                        "nums": nums,
                        "ops": ops,
                        "subgoals": subgoals,
                        "expl_start": expl_start,
                        "expl_end": expl_end,
                    }
                    subgoal_nodes[id] = node

    return subgoal_nodes


def modify_trajectory(
    trajectory: str,
    subgoals_list: List[List[List[str]]],
    depth: int = 1,
    mode: str = "rand",
) -> str:
    """
    Modifies the trajectory to achieve the subgoal at the specified depth.
    """
    if depth == 1:
        # Match root node
        expl_pattern = r"Current State: (\d+):(\[.*?\]), Operations: (\[.*?\])\n"
        expl_match = list(re.finditer(expl_pattern, trajectory))[0]

        # Get node information
        id = "0"
        target = int(expl_match.group(1))
        nums = eval(expl_match.group(2))
        ops = eval(expl_match.group(3))
        expl_start = expl_match.start()
        expl_end = expl_match.end()
        subgoals = random.choice(subgoals_list)

        # Set parent node to root node
        parent_id = id
        parent_node = {
            "target": target,
            "nums": nums,
            "ops": ops,
            "subgoals": subgoals,
            "expl_start": expl_start,
            "expl_end": expl_end,
        }
    else:
        # Get subgoal nodes at previous depth
        subgoal_nodes = get_subgoal_nodes(trajectory, subgoals_list, depth - 1)

        # Return trajectory if no subgoal nodes are found
        if len(subgoal_nodes) == 0:
            return trajectory

        # Set parent node to first subgoal node
        parent_id = list(subgoal_nodes.keys())[0]
        parent_node = subgoal_nodes[parent_id]

    # Get subgoal and operation at current depth
    target = parent_node["target"]
    nums: List[int] = parent_node["nums"]
    subgoal = parent_node["subgoals"][depth]
    op = subgoal[-1]

    # Update numbers
    operand1, operand2, result = parse_operation(op)
    try:
        nums.remove(operand1)
        nums.remove(operand2)
        nums.append(result)
    except:
        return trajectory

    # Match explored child nodes
    parent_expl_end = parent_node["expl_end"]
    expl_pattern = (
        rf"Moving to Node #({parent_id},\d+)\n"
        r"Current State: \d+:\[.*?\], Operations: \[.*?\]\n"
    )
    expl_matches = list(re.finditer(expl_pattern, trajectory[parent_expl_end:]))

    # Get explored child nodes
    expl_child_nodes = {}
    for expl_match in expl_matches:
        # Get node information
        id = expl_match.group(1)
        expl_start = parent_expl_end + expl_match.start()
        expl_end = parent_expl_end + expl_match.end()

        # Check if node has been generated
        if id not in expl_child_nodes:
            gen_pattern = (
                r"Exploring Operation: \d+[-+*/]\d+=\d+, Resulting Numbers: \[.*?\]\n"
                rf"Generated Node #{id}: \d+:\[.*?\] Operation: \d+[-+*/]\d+=\d+\n"
            )
            gen_matches = list(
                re.finditer(gen_pattern, trajectory[parent_expl_end:expl_start])
            )
            if len(gen_matches) == 1:
                gen_match = gen_matches[0]
                gen_start = parent_expl_end + gen_match.start()
                gen_end = parent_expl_end + gen_match.end()
                node = {
                    "expl_start": expl_start,
                    "expl_end": expl_end,
                    "gen_start": gen_start,
                    "gen_end": gen_end,
                }
                expl_child_nodes[id] = node

    # Match generated child nodes
    gen_pattern = (
        rf"Exploring Operation: {op}, Resulting Numbers: {nums}\n"
        rf"Generated Node #({parent_id},\d+): {target}:{nums} Operation: {op}\n"
    )
    gen_matches = list(re.finditer(gen_pattern, trajectory[parent_expl_end:]))

    # Get generated child nodes
    gen_child_nodes = {}
    for gen_match in gen_matches:
        # Get node information
        id = gen_match.group(1)
        gen_start = parent_expl_end + gen_match.start()
        gen_end = parent_expl_end + gen_match.end()

        if id not in gen_child_nodes:
            node = {"gen_start": gen_start, "gen_end": gen_end}
            gen_child_nodes[id] = node

    # If subgoal node is generated but not explored
    if len(gen_child_nodes) > 0:
        # Set child id
        child_id = list(gen_child_nodes.keys())[0]
        child_node = gen_child_nodes[child_id]

        # Add exploration line
        gen_end = child_node["gen_end"]
        expl_lines = (
            f"Moving to Node #{child_id}\n"
            f"Current State: {target}:{nums}, Operations: {subgoal}\n"
        )
        trajectory = trajectory[:gen_end] + expl_lines
    # If no explored child nodes are found
    elif len(expl_child_nodes) == 0:
        # Set child id
        child_id = parent_id + ",0"

        # Add generation and exploration line
        lines = (
            f"Exploring Operation: {op}, Resulting Numbers: {nums}\n"
            f"Generated Node #{child_id}: {target}:{nums} Operation: {op}\n"
            f"Moving to Node #{child_id}\n"
            f"Current State: {target}:{nums}, Operations: {subgoal}\n"
        )
        trajectory = trajectory[:parent_expl_end] + lines
    # Otherwise
    else:
        # Set child id
        if mode == "first":
            child_id = list(expl_child_nodes.keys())[0]
        elif mode == "last":
            child_id = list(expl_child_nodes.keys())[-1]
        elif mode == "rand":
            child_id = random.choice(list(expl_child_nodes.keys()))
        else:
            raise ValueError(f"Invalid mode: {mode}")
        child_node = expl_child_nodes[child_id]

        # Modify exploration line
        expl_start = child_node["expl_start"]
        expl_lines = (
            f"Moving to Node #{child_id}\n"
            f"Current State: {target}:{nums}, Operations: {subgoal}\n"
        )
        trajectory = trajectory[:expl_start] + expl_lines

        # Modify generation line
        gen_start = child_node["gen_start"]
        gen_end = child_node["gen_end"]
        gen_lines = (
            f"Exploring Operation: {op}, Resulting Numbers: {nums}\n"
            f"Generated Node #{child_id}: {target}:{nums} Operation: {op}\n"
        )
        trajectory = trajectory[:gen_start] + gen_lines + trajectory[gen_end:]

    return trajectory


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_length: int = 4096,
    temperature: float = 0.8,
    stop_strings: List[str] = ["Goal Reached", "Exited"],
) -> List[str]:
    """
    Generates search paths starting from the given prompts.
    """
    # Encode prompts
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    tokenizer.padding_side = prev_padding_side
    inputs = inputs.to("cuda")

    # Generate tokens
    if temperature == 0.0:
        all_tokens = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    else:
        all_tokens = model.generate(
            **inputs,
            max_length=max_length,
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
    start = args.start
    end = min(start + args.offset, len(data))
    data = data[start:end]

    # Load results
    if args.depth == 1:
        result_file = f"results_s{args.seed}_{args.split}_{start}_{end}.json"
    elif args.depth == 2:
        result_file = f"results_{args.mode}_s{args.seed}_depth1_{args.split}_{start}_{end}.json"
    else:
        raise ValueError(f"Invalid depth: {args.depth}")
    result_file = os.path.join(args.ckpt, result_file)
    with open(result_file, "r") as f:
        results = json.load(f)
    trajectories = results["trajectories"]
    ratings = results["ratings"]

    # Modify trajectories
    indices = []
    prompts = []
    lengths = []
    for i, (sample, trajectory, rating) in enumerate(zip(data, trajectories, ratings)):
        # Continue if the trajectory is successful
        if rating > 0.0:
            continue

        # Get subgoal nodes
        optimal_path = sample["optimal_path"].strip()
        subgoals_list = get_subgoals_list(optimal_path)
        subgoal_nodes = get_subgoal_nodes(trajectory, subgoals_list, depth=args.depth)

        # Continue if the subgoal nodes are found
        if len(subgoal_nodes) > 0:
            continue

        # Get the new trajectory
        new_trajectory = modify_trajectory(
            trajectory, subgoals_list, depth=args.depth, mode=args.mode
        )

        # Compute the length of the new trajectory
        length = len(tokenizer(new_trajectory)["input_ids"])
        if length >= args.max_length:
            continue

        # Append the new trajectory
        indices.append(i)
        prompts.append(new_trajectory)
        lengths.append(length)

    # Sort by the length of the new trajectories
    combined = list(zip(lengths, indices, prompts))
    combined = sorted(combined)
    lengths, indices, prompts = zip(*combined)

    # Augment trajectories
    for batch_start in trange(0, len(indices), args.batch_size):
        # Generate search paths starting from the new trajectories
        batch_end = min(batch_start + args.batch_size, len(indices))
        indices_batch = indices[batch_start:batch_end]
        prompts_batch = prompts[batch_start:batch_end]
        gen_search_paths = generate(
            model,
            tokenizer,
            prompts_batch,
            max_length=args.max_length,
            temperature=args.temperature,
        )

        # Update trajectories and ratings
        for i, gen_search_path in zip(indices_batch, gen_search_paths):
            gen_rating, _ = metric_fn(gen_search_path)
            trajectories[i] = gen_search_path
            ratings[i] = gen_rating

    # Save results
    results = {"trajectories": trajectories, "ratings": ratings}
    if args.depth == 1:
        result_file = f"results_{args.mode}_s{args.seed}_depth1_{args.split}_{start}_{end}.json"
    elif args.depth == 2:
        result_file = f"results_{args.mode}_s{args.seed}_depth2_{args.split}_{start}_{end}.json"
    else:
        raise ValueError(f"Invalid depth: {args.depth}")
    result_file = os.path.join(args.ckpt, result_file)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mode", default="rand", type=str)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--offset", default=1000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    args = parser.parse_args()

    # Run main
    main(args)
