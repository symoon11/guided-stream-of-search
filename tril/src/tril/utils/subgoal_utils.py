import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer


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
) -> List[Dict[str, Any]]:
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


def compute_subgoal_reward(
    tokenizer: PreTrainedTokenizer,
    actions: torch.Tensor,
    ref_ids: torch.Tensor,
) -> torch.Tensor:
    # Decode actions and references
    gen_texts = tokenizer.batch_decode(actions, skip_special_tokens=True)
    ref_texts = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

    # Compute subgoal reward
    subgoal_rewards = torch.zeros_like(actions).float()
    for i, (action, gen_text, ref_text) in enumerate(
        zip(actions, gen_texts, ref_texts)
    ):
        # Get subgoal nodes
        subgoals_list = get_subgoals_list(ref_text)
        subgoal_nodes = {}
        subgoal_nodes.update(get_subgoal_nodes(gen_text, subgoals_list, depth=1))
        subgoal_nodes.update(get_subgoal_nodes(gen_text, subgoals_list, depth=2))

        # Get position of seperator token
        sep_token = "\n"
        sep_token_id = tokenizer(sep_token)["input_ids"][0]
        sep_token_pos = [match.end() for match in re.finditer("\n", gen_text)]
        sep_token_id_pos = torch.where(action == sep_token_id)[0]

        # Find position of subgoal nodes and update subgoal rewards
        for node_id in subgoal_nodes.keys():
            node = subgoal_nodes[node_id]
            expl_end = node["expl_end"]
            j = sep_token_id_pos[sep_token_pos.index(expl_end)]
            subgoal_rewards[i, j] = 0.2

    return subgoal_rewards
