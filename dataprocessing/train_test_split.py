"""
Shuffle the data and split the data in a training and test set
"""
import argparse
import json
import os
import random
from collections import Counter

# Data directory. Follow structure with default or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../", "data"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)


def main():
    parser = argparse.ArgumentParser(description="Split in train and test")

    parser.add_argument(
        "--input",
        default=os.path.join(
            DATA_DIR, "sentiment/full_data/classification_sentiment.jsonl"
        ),
        help="All data, the data_records to be split. One json record per line",
    )

    parser.add_argument(
        "--output_folder",
        default=os.path.join(DATA_DIR, "sentiment/classification"),
        help="Folder where the split data set will be put",
    )

    parser.add_argument(
        "--label", default="sentiment", help="The label for filtering by count"
    )

    parser.add_argument(
        "--min_count",
        default=1000,
        help="A label which occurs less in the data is not used",
    )

    parser.add_argument(
        "--max_count",
        default=2000000,
        help="Per label - more occurrences of records with a label are ignored",
    )

    args = parser.parse_args()

    # List of arbitrary length, distribute data in buckets (number of buckets len(list))
    # and assign to set designated by the string
    # 0 or "" for skipping the line
    input_split = [
        "train",
        "train",
        "eval",
        "train",
        "train",
        "train",
        "train",
        "eval",
        "train",
        "train",
    ]
    # Small set
    # input_split = ["train", "eval", "train", "test",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # Read the input
    json_list = []
    with open(args.input, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            json_data = json.loads(line)
            json_list.append(json_data)

    random.shuffle(json_list)

    # Count and split the data
    split_data = {}
    label_count = Counter()
    for i, json_data in enumerate(json_list):
        output_set = input_split[i % len(input_split)]
        if isinstance(output_set, str) and len(output_set) > 0:
            label = json_data[args.label]
            # Skip if we have enough of label already
            if label_count[label] > args.max_count:
                continue
            label_count[label] += 1
            if output_set not in split_data:
                split_data[output_set] = []
            split_data[output_set].append(json_data)
        else:
            # skip line
            pass

    input_file_name = os.path.splitext(os.path.basename(args.input))[0]
    for output_set in split_data:
        file_name = os.path.join(
            args.output_folder, "{}_{}.jsonl".format(input_file_name, output_set)
        )
        with open(file_name, "w", encoding="utf-8") as out:
            for json_data in split_data[output_set]:
                if label_count[json_data[args.label]] >= args.min_count:
                    out.write(json.dumps(json_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
