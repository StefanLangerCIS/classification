""" Classifiers based on scikit-learn
    They all have the same interface, so they can be wrapped in one class
    Derived from TextClassifier
"""
import os
import re
from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# Sklearn: Classifiers
from sklearn.tree import DecisionTreeClassifier, export_text

from classification.encode.text_encoder import create_data_table, TextEncoder, TextEncoderSparse
from classification.classify.text_classifier import (ClassifierResult, TextClassifier)
from utils.app_logging import app_logger


class SklearnClassifier(TextClassifier):
    """
    Classify with sklearn classifiers
    """

    supported_classifiers = [
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "LogisticRegression",
        "MLPClassifier",
        "GaussianNB",
        "MultinomialNB",
        "KNeighborsClassifier",
        "LinearSVC",
        "Perceptron",
    ]

    def __init__(
            self,
            classifier_type: str,
            classifier_preprocessor: TextEncoder,
            model_folder_path: str = None
    ):
        """
        Initialize the classifier
        :param classifier_type: The name of the classifiers
        :param model_folder_path: The folder where to store models.
        """
        if model_folder_path:
            self.model_folder_path = model_folder_path
        else:
            model_path = os.path.abspath(__file__)
            model_path = os.path.dirname(model_path)
            model_path = os.path.join(model_path, "data", "models")
            self.model_folder_path = model_path
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        if classifier_type == "KNeighborsClassifier":
            self.sklearn_classifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
        elif classifier_type == "MLPClassifier":
            self.sklearn_classifier = MLPClassifier(
                hidden_layer_sizes=(200,), activation="logistic"
            )
        elif classifier_type == "LinearSVC":
            self.sklearn_classifier = LinearSVC()
        elif classifier_type == "GaussianNB":
            self.sklearn_classifier = GaussianNB()
        elif classifier_type == "MultinomialNB":
            self.sklearn_classifier = MultinomialNB(alpha=0.01)
        elif classifier_type == "LogisticRegression":
            self.sklearn_classifier = LogisticRegression(solver="sag", n_jobs=6)
        elif classifier_type == "RandomForestClassifier":
            self.sklearn_classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif classifier_type == "DecisionTreeClassifier":
            self.sklearn_classifier = DecisionTreeClassifier(
                min_samples_split=10, max_features="sqrt"
            )
        elif classifier_type == "Perceptron":
            self.sklearn_classifier = Perceptron()
        else:
            raise Exception(
                f"Unsupported classifier type {classifier_type}. Use one of {self.supported_classifiers}"
            )

        self.classifier_type = classifier_type
        self.classifier_name = classifier_type
        self.preprocessor = classifier_preprocessor

    def name(self) -> str:
        return self.classifier_name

    def info(self) -> Dict[str, Any]:
        return self.sklearn_classifier.get_params()

    def classify(self, data: dict) -> List[ClassifierResult]:
        """
        Classify a data point.

        :param data: The data dictionary with field 'text'.
        :return: A list with one classifier result
        """
        data_table: pd.DataFrame = create_data_table([data])
        matrix = self.preprocessor.create_matrix(data_table)

        predicted = self.sklearn_classifier.predict(matrix)
        predicted_class = predicted[0]

        result = ClassifierResult(predicted_class, -1, "")
        return [result]

    def train(self, training_data: List[Dict]) -> None:
        """
        Train the classifier
        :param training_data: List of training data points with fields 'text' and 'label'.
        :return: Nothing
        """
        data_train = create_data_table(training_data)
        matrix_train = self.preprocessor.create_matrix(data_train, train=True)
        print(f"INFO: Starting fitting of classifier with {len(training_data)} training points.")
        self.sklearn_classifier.fit(matrix_train, data_train.label)
        print("INFO: Fitting completed")
        self._save_model_information()

    def _save_model_information(self) -> None:
        """
        Print detailed information about the model
        :return: None
        """
        pass
        # if self.classifier_type == "DecisionTreeClassifier":
        #    self.print_decision_tree()
        # else:
        #    pass

    def _print_decision_tree(self):
        """
        Print decision tree rules
        :return:
        """
        rules_text = export_text(self.sklearn_classifier, max_depth=100)
        # Vocabulary for replacement in the data which contains
        # feature numbers only
        if isinstance(self.preprocessor, TextEncoderSparse):
            vocab = self.preprocessor.count_vectorizer.vocabulary_
        else:
            app_logger.warning("Cannot print decision tree.")
            return
        vocabulary = dict((feature, word) for word, feature in vocab.items())
        rules = rules_text.split("\n")
        lines = []
        for rule in rules:
            if "feature_" in rule:
                word_id_str = re.sub(".*feature_([0-9]+).*", r"\1", rule)
                word_id = int(word_id_str)
                if word_id in vocabulary:
                    word = vocabulary[word_id]
                else:
                    word = "UNK"
                rule = rule.replace(f"feature_{word_id_str}", word)
                lines.append(rule)
            else:
                lines.append(rule)

        with open(
                os.path.join(self.model_folder_path, "decision_rules.txt"),
                "w",
                encoding="utf-8",
        ) as out:
            for line in lines:
                out.write(line + "\n")
