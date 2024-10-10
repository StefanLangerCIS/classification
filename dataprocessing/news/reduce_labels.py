"""
Remove and merge some labels in the original Huffington post data set
"""
import argparse
import json
import os
import random
from collections import Counter

# Data directory. Follow structure with default or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../../", "data"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)


def main():
    parser = argparse.ArgumentParser(description="Huffington Post: reduce the labels")

    parser.add_argument(
        "--input",
        default=os.path.join(DATA_DIR, "news/classification_news_original.jsonl"),
        help="All data, the data_records to be split. One json record per line",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "news/classification_news.jsonl"),
        help="Folder where the split data set will be put",
    )

    parser.add_argument(
        "--label", default="category", help="The label for the category to process"
    )

    args = parser.parse_args()

    # Label merges, specific for news data
    label_merges = {
        "PARENTS": "PARENTING",
        "ARTS": "CULTURE & ARTS",
        "ARTS & CULTURE": "CULTURE & ARTS",
        "GREEN": "ENVIRONMENT",
        "FOOD & DRINK": "TASTE",
        "THE WORLDPOST": "WORLD",
        "WORLD NEWS": "WORLD",
        "WORLDPOST": "WORLD",
        "LATINO VOICES": None,
    }

    # Read the input data
    json_list = []
    with open(args.input, encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            json_data = json.loads(line)
            json_list.append(json_data)

    # Filter the list based on the label merges
    filtered_json_list = []
    for item in json_list:
        label = item[args.label]
        if label in label_merges:
            target_label = label_merges[label]
            item[args.label] = target_label
        if item[args.label]:
            filtered_json_list.append(item)

    with open(args.output, "w", encoding="utf-8") as out:
        for json_data in filtered_json_list:
            out.write(json.dumps(json_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
