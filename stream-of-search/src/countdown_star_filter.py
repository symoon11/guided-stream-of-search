import argparse
import json
import os

from countdown_utils import get_target_nums


def main(args):
    # Load results
    prefix = f"{args.prefix}_{args.split}"
    result_files = [f for f in os.listdir(args.ckpt) if f.startswith(prefix)]
    result_files = sorted(result_files)
    trajectories = []
    ratings = []
    for result_file in result_files:
        result_file = os.path.join(args.ckpt, result_file)
        with open(result_file, "r") as f:
            results = json.load(f)
            trajectories += results["trajectories"]
            ratings += results["ratings"]

    # Filter trajectories
    data = []
    for trajectory, rating in zip(trajectories, ratings):
        if rating > 0.0:
            trajectory += "\n"
            target, nums = get_target_nums(trajectory)
            data.append(
                {
                    "nums": nums,
                    "target": target,
                    "solution": ["No solution available"],
                    "rating": rating,
                    "search_path": trajectory,
                    "optimal_path": "No optimal path available",
                    "heuristic": "base-0",
                    "search_type": "base-0",
                }
            )
    print(f"{len(data)}/{len(trajectories)}={len(data)/len(trajectories)}")

    # Save data
    data_file = os.path.join(args.ckpt, f"filtered_{prefix}.json")
    with open(data_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--prefix", default="results", type=str)
    args = parser.parse_args()

    # Run main
    main(args)
