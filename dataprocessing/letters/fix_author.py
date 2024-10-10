import argparse
import glob
import json
import os
import re

"""
Utility script
Fix the author and the year in the Schiller/Goethe letters
"""


def main():
    parser = argparse.ArgumentParser(
        description="Fix the author and year in the Schiller/Goethe letters"
    )
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Div\ltrs\letters\schiller_goethe\json_old\*.json",
        help="The input directory + input file pattern, json files",
    )
    parser.add_argument(
        "--output",
        default=r"D:\ProjectData\Div\ltrs\letters\schiller_goethe\json",
        help="The output directory for the fixed letters in json format",
    )

    args = parser.parse_args()

    input_files = glob.glob(args.input)
    print(
        "INFO: Fixing author in {0} input files in {1}".format(
            len(input_files), args.input
        )
    )
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as input:
            data = input.read()
            try:
                json_data = json.loads(data)
            except:
                print("Invalid json {0}".format(input_file))
                continue
            # Author
            author = "unknown"
            recipient = json_data["recipient"]
            if not recipient:
                recipient = ""
            if recipient.lower() == "goethe":
                recipient = "Johann Wolfgang von Goethe"
                author = "Friedrich Schiller"
            elif recipient.lower() == "schiller":
                author = "Johann Wolfgang von Goethe"
                recipient = "Friedrich Schiller"
            json_data["recipient"] = recipient
            json_data["author"] = author

            # Year:
            date = json_data["date"]
            if not date:
                date = ""
            match = re.search(r"\b1[78][0-9][0-9]\b", date)
            if match:
                json_data["year"] = int(match.group())

        filename_with_extension = os.path.basename(input_file)
        output_file = os.path.join(args.output, filename_with_extension)
        with open(output_file, "w", encoding="utf-8") as output:
            output.write(
                json.dumps(json_data, indent=4, sort_keys=True, ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
