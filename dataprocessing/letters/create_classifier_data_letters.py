import argparse
import glob
import json

"""
Create TSV/JSON data for classification
Splits the letter text into chunks of a certain size and puts each chunk in a separate record
"""


def _get_common_prefix_from_list(string_list):
    common_prefix = ""
    for string in string_list:
        if common_prefix == "":
            common_prefix = string
        else:
            common_prefix = _get_common_prefix(string, common_prefix)
    return common_prefix


def _get_common_prefix(string1, string2):
    length = 0
    for i in range(0, min(len(string1), len(string2))):
        if string1[i] != string2[i]:
            break
        else:
            length += 1

    return string1[0:length]


def main():
    parser = argparse.ArgumentParser(
        description="Script for creating a json file ready for classification"
    )
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Uni\ltrs\letters\*\json\*.json",
        help="The input directory + input file pattern",
    )
    parser.add_argument(
        "--output",
        default=r"D:\ProjectData\Uni\ltrs\classifier\classifier_data.json",
        help="The output directory the classifier data",
    )
    parser.add_argument(
        "--format", choices=["json", "tsv"], default="json", help="Output format"
    )

    args = parser.parse_args()

    input_files = glob.glob(args.input)

    common_prefix = _get_common_prefix_from_list(input_files)

    print(
        "INFO: Creating classifier data from {0} input files in {1}".format(
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
            # Author
            author = json_data["author"]
            year = json_data["year"]
            text = json_data["text"]
            language = json_data["lang"]
            text_parts = text.split(".")
            part = ""
            for i in range(0, len(text_parts)):
                text_part = text_parts[i]
                if len(part) > 0:
                    part += "."
                part += text_part
                if len(part) > 250 or (len(part) > 50 and i + 1 == len(text_parts)):
                    record = {}
                    record["author"] = author
                    record["year"] = year
                    record["lang"] = language
                    record["text"] = part.strip()
                    record["file"] = input_file[len(common_prefix) :].replace("\\", "/")
                    data.append(record)
                    part = ""

    print(
        "INFO: Created  {0} line of data, writing to {1}".format(len(data), args.output)
    )
    with open(args.output, "w", encoding="utf-8") as output:
        if args.format == "json":
            for record in data:
                output.write("{0}\n".format(json.dumps(record, ensure_ascii=False)))
        else:  # tsv
            output.write(
                "{0}\t{1}\t{2}\t{3}\n".format("text", "author", "year", "lang")
            )
            for record in data:
                output.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(
                        record["text"], record["author"], record["year"], record["lang"]
                    )
                )


if __name__ == "__main__":
    main()
