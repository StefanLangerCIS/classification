import abc
import json
import random
from abc import ABC
from typing import Any, Dict, List


class ClassifierResult:
    """
    Generic class for holding a classifier result
    """

    def __init__(self, class_name: str, class_score=-1, meta_information=""):
        """
        :param class_name: can be any string
        :param class_score: any number. Optimally normalized to value between 0...1
        :param meta_information: any text information you would like to add to the result
        """
        self.class_name = class_name
        self.class_score = class_score
        self.meta_information = meta_information

    def __repr__(self):
        return f"{self.__class__} object. class_name: {self.class_name}, class_description: {self.class_score}, meta_information: {self.meta_information}"


class TextClassifier(ABC):
    """
    Abstract base class for ADS classifier + the definition of a classifier result
    """

    @abc.abstractmethod
    def name(self) -> str:
        """
        Return the classifier name.
        :return: Name of the classifier
        """
        return ""

    @abc.abstractmethod
    def info(self) -> Dict[str, Any]:
        """
        Information about the classifier (e.g. hyperparameters)
        :return: Additional info about the classifier
        """
        return {"AbstractClassifier": True}

    @abc.abstractmethod
    def train(self, training_data: List[Dict]) -> None:
        """
        Train the classifier
        :param training_data: List of training data points with fields 'text' and 'label'
        :return: Nothing
        """
        return

    @abc.abstractmethod
    def classify(self, data) -> List[ClassifierResult]:
        """
        Classify a given text
        :param data: dictionary (parsed json) - the record to classify, needs field 'text'
        :return: List of predicted classes (for most classifier, just one class)
        """
        """
        Return an ordered list of ClassifierResults
        """
        return []


def get_data_records_from_file(
    training_file: str, text_label: List[str], class_label: str, mx: int = 0
) -> List[Dict]:
    """
    Retrieve the data records from file (for training)
    """
    with open(training_file, encoding="utf-8") as training_fp:
        data_records = []
        for line in training_fp:
            record = json.loads(line)
            data_record = {}
            data_record["text"] = " ".join([record[x] for x in text_label])
            data_record["label"] = record[class_label]
            data_records.append(data_record)

    if mx is not None and mx > 0:
        data_records = random.sample(data_records, mx)

    return data_records
