import argparse
import json
import os

import pandas as pd

from classification.classify.run_classifier import ClassificationResultData
from classification.env.env import CLASSIFIER_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or several text classifiers"
    )

    parser.add_argument(
        "--input",
        default=CLASSIFIER_OUTPUT_DIR,
        help="Folder where to write the classifier evaluation results",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.join(CLASSIFIER_OUTPUT_DIR, "classifier_results.csv"),
        ),
        help="The CSV output",
    )

    args = parser.parse_args()
    classifier_results = []
    for file in os.listdir(args.input):
        if file.endswith(".json"):
            with open(os.path.join(args.input, file), "r") as f:
                classifier_result = ClassificationResultData.model_validate(json.load(f))
                classifier_results.append({
                    "file": file.replace(".json", ""),
                    "classifier": classifier_result.classifier_name,
                    "class_label": classifier_result.class_label,
                    "text_labels": ", ".join(classifier_result.text_labels),
                    "preprocessing": json.dumps(classifier_result.preprocessing_info),
                    "accuracy": classifier_result.classification_report["accuracy"]
                })

    classifier_results.sort(key=lambda x: x["accuracy"], reverse=True)
    df = pd.DataFrame(classifier_results)

    df.to_csv(args.output, index=False)  # index=False to avoid writing row numbers


if __name__ == "__main__":
    main()
