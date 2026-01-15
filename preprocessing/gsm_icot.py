# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import argparse


def main(split):
    """
    Convert icot text CODI to JSON format.
    Args:
        split (str): The dataset split (e.g., train, test, valid).
    """
    with open(f"CODI/gsm_{split}.txt") as f:
        data = f.readlines()
    data = [
        {
            "question": d.split("||")[0],
            "steps": d.split("||")[1].split("##")[0].strip().split(" "),
            "answer": d.split("##")[-1].strip(),
        }
        for d in data
    ]
    json.dump(data, open(f"CODI/gsm_{split}.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert icot text CODI to JSON format."
    )
    parser.add_argument(
        "split", type=str, help="The dataset split (e.g., train, test, valid)."
    )
    args = parser.parse_args()
    main(args.split)
