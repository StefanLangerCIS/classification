import argparse
import json

import pandas

"""
This script is intended for producing a jsonl file out of a CSV file
"""


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to json")
    parser.add_argument(
        "--input",
        default=r"C:\ProjectData\Uni\classif_srch\data\sentiment\full_data\classification_sentiment.csv",
        help="The file",
    )
    parser.add_argument(
        "--output",
        default=r"C:\ProjectData\Uni\classif_srch\data\sentiment\full_data\classification_sentiment.jsonl",
        help="The output directory for the sentiment data in json format",
    )

    args = parser.parse_args()

    df = pandas.read_csv(args.input)
    json_lines = []
    for row in df.itertuples(index=False):
        row_data = {
            "text": row.review.replace("<br />", "\n"),
            "sentiment": row.sentiment,
        }

        json_lines.append(json.dumps(row_data, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as output:
        for line in json_lines:
            output.write(line + "\n")

    print("INFO: Finished. Created json file {}".format(args.output))


if __name__ == "__main__":
    main()
