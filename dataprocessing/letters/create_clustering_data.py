import argparse
import glob
import json

"""
Write all letters to one file
"""


def main():
    parser = argparse.ArgumentParser(
        description="Script for creating a json file ready for clustering"
    )
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Uni\ltrs\letters\*\json\*.json",
        help="The input directory + input file pattern",
    )
    parser.add_argument(
        "--output",
        default=r"D:\ProjectData\Uni\ltrs\classifier\clustering_data_full_letters.json",
        help="The output directory the clustering data",
    )
    parser.add_argument(
        "--format", choices=["json"], default="json", help="Output format"
    )

    args = parser.parse_args()

    input_files = glob.glob(args.input)

    print(
        "INFO: Creating clustering data from {0} input files in {1}".format(
            len(input_files), args.input
        )
    )
    data = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as input:
            input_data = input.read()
            try:
                json_data = json.loads(input_data)
            except:
                print("Invalid json {0}".format(input_file))
                continue
            data.append(json_data)

    print(
        "INFO: Created  {0} line of data, writing to {1}".format(len(data), args.output)
    )
    with open(args.output, "w", encoding="utf-8") as output:
        if args.format == "json":
            for record in data:
                output.write("{0}\n".format(json.dumps(record, ensure_ascii=False)))


if __name__ == "__main__":
    main()
