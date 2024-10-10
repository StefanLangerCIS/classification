import argparse
import json
import os

"""
Utility script
Repair invalid json files (escape line breaks and tabs within text)
"""


def main():
    parser = argparse.ArgumentParser(description="Repair json")
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Uni\ltrs\data\letters\freud\json",
        help="The input and output directory",
    )

    args = parser.parse_args()
    for dir, dirs, files in os.walk(args.input):
        for filename in files:
            (basename, ext) = os.path.splitext(filename)
            ext = ext.lower()
            if ext == ".txt":
                continue
            filepath = os.path.join(dir, filename)
            with open(filepath, "r", encoding="utf-8") as input:
                json_string = input.read()
            json_string = json_string.replace("\n", " ")
            json_string = json_string.replace("\t", " ")
            try:
                json_data = json.loads(json_string)
            except Exception as ex:
                print(
                    "WARNING: Json error remaining in file after repair {0}: {1}".format(
                        filename, ex
                    )
                )
                json_data = None
            if json_data is not None:
                with open(filepath, "w", encoding="utf-8") as output:
                    json.dump(json_data, output, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
