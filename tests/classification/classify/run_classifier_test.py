import os
import unittest
from typing import Tuple

from classification.classify.run_classifier import run_classifier, ClassificationResultData


def get_test_files() -> Tuple[str, str, str]:
    # get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # build the file path in the test_data subfolder
    train_data = os.path.join(current_dir, "test_data", "classifier_data_train.jsonl")
    test_data = os.path.join(current_dir, "test_data", "classifier_data_test.jsonl")
    test_dir = os.path.join(current_dir, "test_data")
    return train_data, test_data, test_dir


class TestRunClassifier(unittest.TestCase):
    def test_run_classifier(self):
        classifier_type = "LogisticRegression"
        training_data, test_data, test_dir = get_test_files()
        class_label = "lang"
        text_labels = ["text"]
        dense = False
        lsa = False
        output = test_dir
        max_train = 0
        report = run_classifier(
            classifier_type,
            training_data,
            test_data,
            class_label,
            text_labels,
            dense,
            lsa,
            output,
            max_train,
            True,
        )
        self.assertTrue(isinstance(report, ClassificationResultData))
        self.assertTrue(report.classification_report["accuracy"] > 0.5)


if __name__ == "__main__":
    unittest.main()
