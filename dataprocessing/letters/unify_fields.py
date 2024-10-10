import argparse
import glob
import json

"""
Some repairs, make all letters uniform
"""


def main():
    parser = argparse.ArgumentParser(
        description="Script for reparing json format letters from input data in various formats (e.g. epub)"
    )
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Div\ltrs\data\letters\*\json\*.json",
        help="The input directory + input file pattern",
    )

    args = parser.parse_args()

    letters = []
    input_files = glob.glob(args.input)
    print(
        "INFO: Fixing data in {0} input files in {1}".format(
            len(input_files), args.input
        )
    )
    valid_fields = {}
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as input:
            data = input.read()
            try:
                json_data = json.loads(data)
            except:
                print("Invalid json {0}".format(input_file))
                continue
            for field in json_data:
                valid_fields[field] = True

    # To be deleted from all records
    # del valid_fields["recipient_original"]
    # del valid_fields["author_original"]
    # for field in valid_fields:
    #    print(field)
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as input:
            data = input.read()
            try:
                json_data = json.loads(data)
            except:
                print("Invalid json {0}".format(input_file))
                continue
            to_delete = []
            for field in json_data:
                if not field in valid_fields:
                    to_delete.append(field)
                else:
                    if not json_data[field]:
                        json_data[field] = "unknown"
                    json_data[field] = str(json_data[field])
            for field in to_delete:
                del json_data[field]
            for field in valid_fields:
                if not field in json_data:
                    json_data[field] = "unknown"

            if json_data["datetime"] != "unknown" and json_data["date"] == "unknown":
                json_data["date"] = json_data["datetime"]
            del json_data["datetime"]

        with open(input_file, "w", encoding="utf-8") as output:
            output.write(
                json.dumps(json_data, indent=4, sort_keys=True, ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
