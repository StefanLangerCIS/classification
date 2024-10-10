import argparse
import glob
import json
import os
import re

from bs4 import BeautifulSoup

"""
This script is intended for producing files with single letters in json format,
based on letter collections in various input formats (epub, plain text)
"""


class LetterSource2Json:
    """
    Generic class. Base specific letter extraction classes on this one
    """

    def __init__(self, input_file, author=None):
        self.input_file = input_file
        self.author = author
        # Data specific step - define in subclass
        self._load_inputfile()

    def _normalize4json(self, text):
        """
        Normalize the text to be compliant with json
        """
        text = text.replace("\n", " ")
        text = text.replace("\\", " ")
        text = re.sub("[\x00-\x1f\x7f-\x9f]", " ", text)
        text = text.strip()
        text = text.replace('"', "'")
        text = re.sub("  +", " ", text)
        return text

    def _init_meta_information(self):
        """
        Initialize letter meta information to the default/unknown value
        """
        meta = {}
        meta["datetime"] = "unknown"
        meta["format"] = "unknown"
        meta["place"] = "unknown"
        meta["source"] = "unknown"
        meta["author"] = "unknown"
        meta["year"] = None
        if self.author:
            meta["author"] = self.author
        else:
            meta["author"] = "unknown"
        return meta

    def get_letters_json(self):
        """
        Produce the json for the extracted data.
        Whereas the parsing of the data is specific for each input file,
        this step is generic and independent from the input format
        """
        # Data specific step - define in subclass
        data_list = self._get_letters_data()
        json_list = []
        for data in data_list:
            json_attributes = []
            for key, value in data.items():
                if value == None:
                    json_attributes.append('\n    "{0}" : null'.format(key))
                elif isinstance(value, int):
                    json_attributes.append('\n    "{0}" : {1}'.format(key, value))
                else:
                    json_attributes.append('\n    "{0}" : "{1}"'.format(key, value))
            attributes_string = ",".join(json_attributes)
            letter_json = "{{{0}\n}}\n".format(attributes_string)
            try:
                json.loads(letter_json)
                json_list.append(letter_json)
            except Exception as ex:
                print("Error : Could not parse json: {0}".format(ex))
                print(letter_json)

        return json_list

    def _load_inputfile(self):
        raise NotImplementedError(
            "This method needs to be implemented in the derived class"
        )

    def _get_letters_data(self):
        raise NotImplementedError(
            "This method needs to be implemented in the derived class"
        )


class Epub2Json(LetterSource2Json):
    """
    Base class for parsing the Epub format
    """

    def _load_inputfile(self):
        with open(self.input_file, "r", encoding="utf-8") as input_html:
            html = input_html.read()
            html = self._clean_html(html)
            self.parsed_html = BeautifulSoup(html, "html.parser")

    def _clean_html(self, html):
        html = html.replace("[&#57596;]", " ")
        return html

    def _get_letters_data(self):
        raise NotImplementedError(
            "This method needs to be implemented in the derived class"
        )


class Epub2JsonVirgiaWoolf(Epub2Json):
    """
    Class for parsing the Epub format of Virginia Woolfs letters
    Not generic. For other epub letter formats, this needs to be adapted
    """

    def _get_letters_data(self):
        data_list = []
        for heading_tag in ["h3"]:
            headings = self.parsed_html.find_all(heading_tag)
            for heading in headings:
                if re.search("[0-9].* To ", heading.text) is not None:
                    title = self._normalize4json(heading.text)
                    recipient = re.sub("^.* To (.*)$", "\\1", title)
                    text_list = []
                    current_element = heading
                    letter_content_elements = []
                    while True:
                        current_element = current_element.find_next_sibling()
                        if (
                            not current_element
                            or current_element.name[0].lower() == "h"
                        ):
                            break
                        letter_content_elements.append(current_element)
                    data = self._get_meta_information(letter_content_elements)
                    data["recipient"] = recipient
                    data["title"] = title
                    data["text"] = self._get_text(letter_content_elements)
                    data_list.append(data)
        return data_list

    def _get_meta_information(self, letter_content_elements):
        meta = self._init_meta_information()
        for element in letter_content_elements:
            pclass = None
            text = element.text.strip()
            text = self._normalize4json(text)
            if "class" in element.attrs:
                pclass = element["class"][0]
            else:
                continue
            if pclass == "time" and text != "[n.d.]":
                meta["datetime"] = text
                match = re.search(r"\b1[89][0-4][0-9]\b", text)
                if match:
                    meta["year"] = int(match.group())

            # date is a misnomer - this contains information about the source
            elif pclass == "date":
                meta["source"] = text
            # again a misnomer - format of the letter (e.g. typewritten)
            elif pclass == "card":
                meta["format"] = text
            elif pclass == "place" and text != "[n.a.]":
                meta["place"] = text

        return meta

    def _get_text(self, letter_content_elements):
        text_parts = []
        for element in letter_content_elements:
            pclass = ""
            text = element.text.strip()
            if "class" in element.attrs:
                pclass = element["class"][0]
            if not pclass in ["date", "card", "link"]:
                text_parts.append(text)

        text = self._normalize4json(" ".join(text_parts))
        return text


class Epub2JsonHenrikIbsen(Epub2Json):
    """
    Class for parsing the Epub format of Ibsen's letters
    Not generic. For other epub letter formats, this needs to be adapted
    """

    def _get_letters_data(self):
        data_list = []
        letter_sections = self.parsed_html.find_all("div", "letterContainer")
        # letter_sections = self.parsed_html.find_all("//body")
        for letter_section in letter_sections:
            data = self._get_meta_information(letter_section)
            data["text"] = self._get_text(letter_section)
            data_list.append(data)
        return data_list

    def _get_meta_information(self, letter_section):
        meta = self._init_meta_information()
        letter_identifier = letter_section.find("div", "letterIdentifier")
        if letter_identifier:
            match = re.search(r"Til ([^,]+).*", letter_identifier.text)
            if match:
                meta["recipient"] = match.group(1)
                # print(meta["recipient"])

            match = re.search(
                r", ([^0-9]+)", re.sub(r"[\[\]]", " ", letter_identifier.text)
            )
            if match:
                meta["place"] = match.group(1).strip()

            match = re.search(r"\b1[89][0-9][0-9]\b", letter_identifier.text)
            if match:
                meta["year"] = int(match.group())
        return meta

    def _get_text(self, letter_section):
        text_parts = []
        for element in letter_section.find_all("div"):
            if "class" in element.attrs:
                divclass = element["class"][0]
                if divclass in [
                    "dateline",
                    "salute",
                    "paragraph",
                    "signed",
                    "postscript",
                ]:
                    text_parts.append(element.text)

        text = self._normalize4json(r" ".join(text_parts))
        return text


class Epub2JsonJoyce(Epub2Json):
    """
    Class for parsing the Epub format of Joyce's letters
    Not generic. For other epub letter formats, this needs to be adapted
    """

    def _get_letters_data(self):
        data_list = []
        years = self.parsed_html.find_all("h3")
        for year_tag in years:
            year = re.sub(".*(1[89][0-9][0-9]).*", r"\1", year_tag.text)
            print("Processing year: " + str(year))
            letter_section = year_tag.find_next_sibling()
            while letter_section is not None and letter_section.name != "h3":
                if letter_section.name == "h5" and letter_section.text[0:2] == "To":
                    data = self._get_meta_information(letter_section)
                    data["year"] = year
                    data["text"] = self._get_text(letter_section)
                    if len(data["text"]) < 10:
                        print("WARNING " + str(data))
                    data_list.append(data)
                letter_section = letter_section.find_next_sibling()
        return data_list

    def _get_meta_information(self, letter_section):
        meta = self._init_meta_information()
        meta["recipient"] = self._normalize4json(
            re.sub("To *([^,]+).*", r"\1", letter_section.text)
        )
        meta["datetime"] = self._normalize4json(
            re.sub(".*,(.*)", r"\1", letter_section.text)
        )
        letter_section = letter_section.find_next_sibling()
        while letter_section is not None and (
            letter_section.name == "p" or letter_section.name == "div"
        ):
            pclass = None
            if "class" in letter_section.attrs:
                pclass = letter_section["class"][0]
            if pclass == "source":
                meta["source"] = letter_section.text
            elif pclass == "addr":
                meta["place"] = self._normalize4json(letter_section.text)
            letter_section = letter_section.find_next_sibling()
        return meta

    def _get_text(self, letter_section):
        text_parts = []
        letter_section = letter_section.find_next_sibling()
        while letter_section is not None and (
            letter_section.name == "p" or letter_section.name == "div"
        ):
            pclass = None
            if "class" in letter_section.attrs:
                pclass = letter_section["class"][0]
            if pclass is None or pclass == "bye" or pclass == "hangind":
                text_parts.append(letter_section.text)
            if pclass == "stanza":
                text_parts.append(letter_section.text)
                # print(letter_section.text)
            letter_section = letter_section.find_next_sibling()
        text = self._normalize4json(r" ".join(text_parts))
        return text


class PlainText2Json(LetterSource2Json):
    """
    Class for parsing the plain text format of Kafka letters
    (plain text in turn extracted from PDF with Tika)
    Not generic. For other letter formats, this needs to be adapted
    """

    def _load_inputfile(self):
        with open(self.input_file, "r", encoding="utf-8") as input_text:
            self.text_lines = []
            for line in input_text:
                line = self._clean_text(line)
                self.text_lines.append(line)

    def _get_letters_data(self):
        data_list = []
        letter_lines = []
        for line in self.text_lines:
            if (
                line.find("An ") is 0
                and len(letter_lines) > 0
                and letter_lines[-1] == ""
            ):
                if len(letter_lines) > 3:
                    letter_data = self._process_letter(letter_lines)
                    data_list.append(letter_data)
                    # Reset, letter has been processed
                    letter_lines = []
            # Add line to current letter
            letter_lines.append(line)
        return data_list

    def _clean_text(self, text):
        text = re.sub("[\x00-\x1f\x7f-\x9f]", " ", text)
        text = text.strip()
        return text

    def _process_letter(self, letter):
        data = self._init_meta_information()  #
        # print(letter[0])
        data["recipient"] = re.sub("^An ", "", letter[0])
        header = ""
        content = ""
        i = 1
        text_start = 0
        for i in range(1, 10):
            if letter[i] == "" or i >= len(letter):
                text_start = i + 1
                break
            header = header + " " + letter[i]
        text = ""
        for i in range(text_start, len(letter)):
            text = text + " " + letter[i]
        data["text"] = self._normalize4json(text)

        match = re.search(r"\b1[9][0-4][0-9]\b", header)
        if match:
            data["year"] = int(match.group())
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Script for creating json format letters from input data in various formats (e.g. epub)"
    )
    parser.add_argument(
        "--input",
        default=r"D:\ProjectData\Uni\ltrs_original_data\joyce\OEBPS\Text\let*.htm",
        # default = r'D:\ProjectData\Div\ltrs\letters\woolf\xhtml',
        help="The input directory + input file pattern",
    )
    parser.add_argument(
        "--output",
        default=r"D:\ProjectData\Uni\ltrs\data\letters\joyce\json",
        help="The output directory for the letters in json format",
    )
    parser.add_argument(
        "--author", default=r"James Joyce", help="The name of the author of the letters"
    )

    parser.add_argument(
        "--format",
        choices=["xml_vw", "xml_jj", "xml_hi", "text"],
        default="xml_jj",
        help="Format of the input file(s) - xml_vw (epub, Virginia Woolf), xml_jj (epub, Joyce), xml_hi (epub, Henrik Ibsen) or plain text",
    )

    args = parser.parse_args()

    letters = []
    input_files = glob.glob(args.input)
    print(
        "INFO: Extracting letters from {0} input files in {1}".format(
            len(input_files), args.input
        )
    )
    for inputfile in input_files:
        if args.format == "xml_vw":
            input2json = Epub2JsonVirgiaWoolf(inputfile, args.author)
        elif args.format == "xml_hi":
            input2json = Epub2JsonHenrikIbsen(inputfile, args.author)
        elif args.format == "xml_jj":
            input2json = Epub2JsonJoyce(inputfile, args.author)
        else:
            input2json = PlainText2Json(inputfile, args.author)
        letters += input2json.get_letters_json()

    print("INFO: Found {0} letters in {1}".format(len(letters), args.input))

    # Remove existing files first
    files = glob.glob(os.path.join(args.output, "*"))
    print(
        "INFO: Removing {0} files in output directory {1}".format(
            len(files), args.output
        )
    )
    for file in files:
        os.remove(file)

    # Write the json files, each containing one letter, to the output directory
    print("INFO: Writing letters to {0}".format(args.output))
    for i in range(0, len(letters)):
        outputfile = os.path.join(args.output, "letter_{0}.json".format(i + 1))
        with open(outputfile, "w", encoding="utf-8") as output:
            output.write(letters[i])

    print(
        "INFO: Finished. Created {0} json files in {1}".format(
            len(letters), args.output
        )
    )


if __name__ == "__main__":
    main()
