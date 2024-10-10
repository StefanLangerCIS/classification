import argparse
import glob
import json
import os

from langdetect import detect

"""
Utility script
Add the language to a letter in a half-automated way
"""


def main():
    parser = argparse.ArgumentParser(description="Add language")
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Uni\ltrs\data\letters\*\json\*.json",
        help="The input and output directory",
    )

    mapping = {
        "Virginia Woolf": ["en"],
        "Henrik Ibsen": ["da", "de"],
        "James Joyce": ["en", "fr", "it", "de"],
        "Johann Wolfgang von Goethe": ["de"],
        "Friedrich Schiller": ["de"],
        "Wilhelm Busch": ["de"],
        "Антон Чехов": ["ru"],
        "Demosthenes": ["el"],
        "Avshalom Feinberg": ["he"],
        "Sigmund Freud": ["de"],
        "Franz Kafka": ["de"],
        "Петр Чаадаев": ["ru"],
        "Voltaire": ["fr"],
        "Fédéric, Le Prince Royal/Roi de Prusse": ["fr"],
    }
    args = parser.parse_args()
    input_files = glob.glob(args.input)
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as input_fp:
            json_string = input_fp.read()
            json_data = None
            try:
                json_data = json.loads(json_string)
            except Exception as ex:
                print("Json error in file {0}: {1}".format(input_file, ex))
                json_data = None
                continue
        if json_data is None:
            continue
        # Already processed
        if (
            "lang" in json_data
            and json_data["lang"] != "unknown"
            and json_data["lang"] != "no"
        ):
            continue
        author = json_data["author"]
        text = json_data["text"]
        # Old info, not needed any longer
        if "detected_language" in json_data:
            del json_data["detected_language"]
        if "language" in json_data:
            del json_data["language"]
        json_data["lang"] = "unknown"
        if author in mapping:
            expected_langs = mapping[author]
        else:
            expected_langs = []
            print("WARNING: Author {0} not in mapping\n".format(author))
        confirmed_lang = "unknown"
        if len(text) < 100:
            pass
        else:
            lang = detect(text)
            if lang in expected_langs:
                confirmed_lang = lang
            else:
                print(
                    "\n{0},{1}: --- {2}\n\n>>>LANGUAGE:{3}\n".format(
                        os.path.basename(input_file), author, text[0:200], lang
                    )
                )
                reply = ""
                while len(reply) == 0:
                    reply = input("Correct? Y/N/<LANG>\n")
                    reply = reply.lower().strip()
                if reply == "y":
                    confirmed_lang = lang
                elif len(reply) == 2:
                    confirmed_lang = reply
        json_data["lang"] = confirmed_lang
        with open(input_file, "w", encoding="utf-8") as output_fp:
            json.dump(json_data, output_fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
