import argparse
import json
import os

import numpy as np


def main(args):
    # Load data
    prefix = f"stats_final_{args.split}"
    files = sorted([f for f in os.listdir(args.ckpt) if f.startswith(prefix)])
    gen_ratings = []
    ref_ratings = []
    for file in files:
        file = os.path.join(args.ckpt, file)
        with open(file, "r") as f:
            data = json.load(f)
            gen_ratings += data["gen_ratings"]
            ref_ratings += data["ref_ratings"]

    # Compute stats
    num_data = len(ref_ratings)
    gen_rating = np.mean(gen_ratings)
    ref_rating = np.mean(ref_ratings)
    gen_accuracy = np.mean([r > 0 for r in gen_ratings])
    ref_accuracy = np.mean([r > 0 for r in ref_ratings])

    # Print stats
    print(f"Num data: {num_data}")
    print(f"Gen Rating: {gen_rating}, Gen Accuracy: {gen_accuracy}")
    print(f"Ref Rating: {ref_rating}, Ref Accuracy: {ref_accuracy}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()

    # Run main
    main(args)
