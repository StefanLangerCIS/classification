"""
Evaluate any of the classifiers, print a confusion matrix and create further evaluation metrics
"""
import argparse
import os

from sklearn_gridsearch import SklearnGridSearch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or several text classifiers"
    )

    # All available classifier types
    classifier_types = list(SklearnGridSearch.grid.keys())

    base_directory = r"C:\ProjectData\Uni\classif_srch\data\letters\classification"
    parser.add_argument(
        "--training",
        default=os.path.join(base_directory, "classifier_data_train.jsonl"),
        help='The training data for the classifier. If "None", the existing model is loaded (if it exists)',
    )

    parser.add_argument(
        "--input",
        default=os.path.join(base_directory, "classifier_data_test.jsonl"),
        help="The text data to use for evaluation (one json per line)",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(base_directory, "results"),
        help="Folder where to write the evaluation results",
    )

    parser.add_argument(
        "--classifier",
        choices=classifier_types + ["all"],
        default="RandomForestClassifier",
        help="The classifier to use. If 'all' iterate through all available classifiers",
    )

    parser.add_argument(
        "--text_label",
        default="text",
        help="Label/field in the json data contains the text to classify",
    )

    parser.add_argument(
        "--label",
        default="author",
        help="Label/field to use for training and classification",
    )

    parser.add_argument("--verbose", action="store_true", help="Provide verbose output")

    args = parser.parse_args()

    # Run all classifiers
    if args.classifier == "all":
        classifiers = classifier_types
    else:
        classifiers = [args.classifier]

    print(f"INFO: Evaluating classifier(s) hyperparameters {classifiers}")

    # Iterate over the classifiers
    for classifier_type in classifiers:
        parameter_grid = SklearnGridSearch.grid[classifier_type]
        grid_search = SklearnGridSearch(classifier_type, parameter_grid)
        print(f"INFO: grid evaluating {classifier_type}")
        res = grid_search.grid_search(args.training, args.text_label, args.label)
        outfile_name = os.path.join(
            args.output, f"gridsearch_results_{classifier_type}.txt"
        )
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write(f"Classifier: {classifier_type}\n")
            for key in res.keys():
                outfile.write(f"Parameter {key}: {res[key]}\n")


if __name__ == "__main__":
    main()
