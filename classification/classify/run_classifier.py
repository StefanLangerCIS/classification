"""
Evaluate any of the classifiers, print a confusion matrix and create further evalution metrics
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn.exceptions
import sklearn.metrics
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from classification.encode.text_encoder import TextEncoderDense, TextEncoderSparse
from classification.classify.sklearn_classifiers import SklearnClassifier
from classification.classify.text_classifier import get_data_records_from_file

from classification.env.env import CLASSIFIER_TRAINING_FILE, CLASSIFIER_EVALUATION_FILE, CLASSIFIER_OUTPUT_DIR
from utils.app_logging import app_logger


def plot_and_store_confusion_matrix(
        y_true: list,
        y_pred: list,
        file_name: str,
        normalize=True,
        cmap=plt.cm.Blues,
        show=False,
) -> None:
    """
    This function prints and plots the confusion matrix, and saves it to a file
    :param y_true: The true classes
    :param y_pred: The predicted classes
    :param file_name: The file name to store the image of the confusion matrix
    :param normalize: normalize numbers (counts to relative counts)
    :param cmap: Layout
    :param show: Display the matrix. If false, only store it
    :return: Nothing
    """
    np.set_printoptions(precision=2)
    if normalize:
        title = "Normalized confusion matrix"
    else:
        title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=[20, 27])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(file_name)
    if show:
        plt.show()


def count_lines(file: str) -> int:
    # Determine the number of training lines (for the record)
    n_lines = 0
    if file is not None:
        with open(file, encoding="utf-8") as f:
            for _ in f:
                n_lines += 1
    return n_lines


class ClassificationResultData(BaseModel):
    classifier_name: str
    classifier_info: Dict[str, Any]
    class_label: str
    text_labels: List[str]
    is_dense: bool
    n_training: int
    n_classified: int
    training_time: float
    classification_time: float
    n_expected_classes: int
    n_predicted_classes: int
    preprocessing_info: Dict[str, Any]
    classification_report: Dict


def classification_result_to_text(report: ClassificationResultData, file_path) -> None:
    app_logger.info(f"Writing results of classifier {report.classifier_name} to {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("#Info:\n")
        f.write(f"Classifier: {report.classifier_name}\n")
        f.write(f"Parameters: {json.dumps(report.classifier_info)}\n")
        f.write(f"Label: {report.class_label}\n")
        f.write(f"Text labels: {', '.join(report.text_labels)}\n")
        f.write(f"Dense: {report.is_dense}\n")
        f.write("\n#Counts:\n")
        f.write(f"Number of training data_records: {report.n_training}\n")
        f.write(f"Number of classified data_records: {report.n_predicted_classes}\n")
        f.write(
            f"Number of unique classes in data_records: {report.n_expected_classes}\n"
        )
        f.write(
            f"Number of unique classes found: {report.n_predicted_classes}\n"
        )
        f.write("\n#Performance:\n")
        f.write(f"Seconds used for training: {report.training_time}\n")
        f.write(f"Seconds used for classification: {report.classification_time}\n")

        warnings.filterwarnings(
            "ignore", category=sklearn.exceptions.UndefinedMetricWarning
        )

        f.write(f"\n#Classification report:\n{report.classification_report}\n")


def classification_result_to_json(report: ClassificationResultData, file_path) -> None:
    result_dict = {**report.model_dump()}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)


def run_classifier(
        classifier_type: str,
        training_data_file: str,
        input_file: str,
        class_label: str,
        text_labels: List[str],
        dense: bool,
        lsa: bool,
        output_folder: str,
        max_train: int = 0,
        verbose: bool = False,
) -> ClassificationResultData | None:
    """
    Run classifier of classifier_type

    :param classifier_type: The classifier algorithm
    :param training_data_file: jsonl training data. If None, the input file will be split training/test.
    :param input_file: jsonl data to classify
    :param class_label: The class label in the training data
    :param text_labels: The keys in the json which point to the text
    :param dense: Use dense vectors (embeddings)
    :param output_folder: Output folder for classification report
    :param max_train: Maximum number of lines to use for training
    :param verbose: Verbose output
    :return: The classification report as dictionary or None if classification fails.
    """

    if dense:
        classifier_encoder = TextEncoderDense()
    elif lsa:
        classifier_encoder = TextEncoderSparse(lsa=True)
    else:
        classifier_encoder = TextEncoderSparse(lsa=False)

    classifier = SklearnClassifier(classifier_type, classifier_encoder)

    app_logger.info(f"Evaluating classification of classifier {classifier.name()}")

    if training_data_file is not None:
        training_data = get_data_records_from_file(
            training_data_file, text_labels, class_label, max_train
        )

        data_to_classify = get_data_records_from_file(input_file, text_labels, class_label)
    else:
        data = get_data_records_from_file(input_file, text_labels, class_label)
        training_data, data_to_classify = train_test_split(data, test_size=0.2)

    app_logger.info(f"Read {len(training_data)} training records")

    training_time = time.time()
    app_logger.info(f"Training classifier {classifier_type} with text fields {text_labels} for label {class_label}")
    try:
        classifier.train(training_data)
    except Exception as e:
        app_logger.error(f"Training classifier {classifier.name()} failed: {e}")
        return None

    training_time = int(time.time() - training_time)
    app_logger.info(f"Training completed in {training_time} seconds")

    classifier.verbose = verbose
    app_logger.info(f"Starting classification of data with classifier {classifier.name()}")
    predicted_classes = []
    expected_classes = []
    # Keep track of time used
    classification_time = time.time()
    for json_data in data_to_classify:
        res = classifier.classify(json_data)
        class_name = "none"
        if len(res) > 0:
            class_name = res[0].class_name
        predicted_classes.append(class_name)
        expected_classes.append(json_data["label"])

    classification_time = int(time.time() - classification_time)
    app_logger.info(f"Classification completed in {classification_time} seconds")

    classification_report = sklearn.metrics.classification_report(expected_classes, predicted_classes, output_dict=True)
    classification_result = ClassificationResultData(
        classifier_name=classifier.name(),
        classifier_info=classifier.info(),
        class_label=class_label,
        text_labels=text_labels,
        is_dense=dense,
        n_training=len(training_data),
        n_classified=len(data_to_classify),
        training_time=training_time,
        classification_time=classification_time,
        n_expected_classes=len(set(expected_classes)),
        n_predicted_classes=len(set(predicted_classes)),
        preprocessing_info=classifier_encoder.info(),
        classification_report=classification_report,
    )

    outfile_identifier = f"results_{classifier.name()}_{class_label}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    outfile_name = os.path.join(output_folder, f"{outfile_identifier}.txt")
    classification_result_to_text(classification_result, outfile_name)

    outfile_json = os.path.join(output_folder, f"{outfile_identifier}.json")
    classification_result_to_json(classification_result, outfile_json)

    # Also store confusion matrix as image
    imagefile_name = os.path.join(output_folder, f"{outfile_identifier}.jpg")
    plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)
    app_logger.info(f"Processing completed for classifier {classifier.name()}")
    return classification_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or several text classifiers"
    )

    # All available classifier types
    classifier_types = SklearnClassifier.supported_classifiers

    parser.add_argument(
        "--training",
        default=CLASSIFIER_TRAINING_FILE,
        help='The training data for the classifier. If "None", we do a train-test-split on the input.',
    )

    parser.add_argument(
        "--input",
        default=os.path.join(
            CLASSIFIER_EVALUATION_FILE,
        ),
        help="The text data to use for evaluation (one json per line)",
    )

    parser.add_argument(
        "--output",
        default=CLASSIFIER_OUTPUT_DIR,
        help="Folder where to write the classifier evaluation results",
    )

    parser.add_argument(
        "--classifier",
        choices=classifier_types + ["all"],
        default="all",
        help="The classifier to use. If 'all' iterate through all available classifiers",
    )

    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use transformer dense vector for classification",
    )

    parser.add_argument(
        "--text_label",
        default="body",
        help="Label/field in the json data which contains the text to classify."
             "Can also be multiple labels separated by comma (,)",
    )

    parser.add_argument(
        "--label",
        default="superclass",
        help="Label/field to use for training and classification",
    )

    parser.add_argument(
        "--max_train",
        type=int,
        default=0,
        help="Maximum number of data points to use for training",
    )

    parser.add_argument("--verbose", action="store_true", help="Provide verbose output")

    args = parser.parse_args()

    # Run all classifiers
    if args.classifier == "all":
        classifiers = classifier_types
    else:
        classifiers = [args.classifier]

    app_logger.info(f"Evaluating classifier(s) {classifiers}")

    # Iterate over the classifiers
    text_labels = args.text_label.split(",")
    for classifier_type in classifiers:
        run_classifier(
            classifier_type,
            args.training,
            args.input,
            args.label,
            text_labels,
            args.dense,
            args.output,
            args.max_train,
            args.verbose,
        )


if __name__ == "__main__":
    main()
