""" Grid search implementation
"""

from typing import Dict

import pandas as pd
# Sklearn: Classifiers
from sklearn.ensemble import RandomForestClassifier
# Sklearn: Other utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from classification.classify.text_classifier import get_data_records_from_file


class SklearnGridSearch:
    """
    Run grid search with selected sklearn classifiers
    """

    grid = {
        "RandomForestClassifier": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 20, 40, 100, 200],
            "min_samples_leaf": [1, 2, 4, 8],
        },
        "MLPClassifier": {
            "activation": ["logistic", "relu"],
            "hidden_layer_sizes": [(10,), (20,), (50,)],
        },
    }

    def __init__(self, classifier_type: str, parameters: Dict, verbose: bool = False):
        """
        Initialize the grid search.

        :param classifier_type: The name of the classifier algorithm.
        :param parameters: The parameter grid to search over.
        :param verbose: Enable verbose output.
        """
        self.verbose = verbose

        if classifier_type not in self.grid:
            raise Exception(
                f"Unsupported classifier type {classifier_type}. Use one of {self.grid.keys()}"
            )

        if classifier_type == "RandomForestClassifier":
            self.sklearn_classifier = RandomForestClassifier()
        elif classifier_type == "MLPClassifier":
            self.sklearn_classifier = MLPClassifier()

        self.parameters = parameters

        self.classifier_type = classifier_type
        self.count_vectorizer = CountVectorizer(
            min_df=10, max_df=0.8, ngram_range=(1, 1)
        )
        self.tfidf_transformer = TfidfTransformer(use_idf=True)

    def grid_search(
        self, training_data: str, text_label: str, class_label: str
    ) -> Dict:
        """
        Run grid search to find the best hyperparameters.

        :param training_data: File name. Training data is one json per line.
        :param text_label: JSON field which contains the text.
        :param class_label: JSON field which contains the class label.
        :return: The best parameters found by grid search.
        """

        data_train = get_data_records_from_file(
            training_data, [text_label], class_label, 10000
        )
        print(f"INFO: grid evaluation with {len(data_train)} data points")
        data_train = pd.DataFrame.from_dict(data_train)
        data_train.fillna("", inplace=True)

        matrix_train_counts = self.count_vectorizer.fit_transform(data_train.text)
        self.tfidf_transformer = self.tfidf_transformer.fit(matrix_train_counts)
        matrix_train_tf = self.tfidf_transformer.transform(matrix_train_counts)
        matrix_train_tf = matrix_train_tf.toarray()

        grid_search = GridSearchCV(self.sklearn_classifier, self.parameters, n_jobs=10)
        grid_search.fit(matrix_train_tf, data_train.label)
        print(grid_search.best_params_)
        return grid_search.best_params_
