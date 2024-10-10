"""
Correlation between two nominal labels
"""
import argparse
import glob
import json
import os
from typing import List, Dict
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
from scipy.stats import chi2_contingency

# Data directory. Follow structure in _default_data_dir or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../", "data", "news", "clustering"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)

def calculate_correlation(data: List[Dict], label1: str, label2: str, data_name=""):
    df = pd.DataFrame(data)
    print(df)

    # create a contingency table
    contingency_table = pd.crosstab(df[label1], df[label2])

    # perform chi-square test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-square value is {chi2}")
    print(f"P-value is {p}")
    print(f"Degrees of freedom is {dof}")
    print("Expected contingency table:")
    print(expected)

    # Visualizing the contingency table
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap="YlGnBu", yticklabels=True)
    plt.title(f"Contingency Table for algorithm {data_name}")
    plt.show()
    return ""

def main():
    parser = argparse.ArgumentParser(
        description="Calculate correlation between nominal features"
    )

    parser.add_argument(
        "--input",
        default=os.path.join(
            DATA_DIR, "results", "clusters*.jsonl"
        ),
        help="Data correlation",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "results"),
        help="Folder where to write the info",
    )

    parser.add_argument(
        "--label1",
        default="category",
        help="First variable",
    )

    parser.add_argument(
        "--label2",
        default="cluster_id",
        help="First variable",
    )

    args = parser.parse_args()

    input_files = glob.glob(args.input)

    for input_file in input_files:

        print(f"INFO: Running correlation {input_file}")
        records = []
        with open(input_file, encoding="utf-8") as input_fp:
            for line in input_fp:
                record = json.loads(line)
                records.append(record)

        print(f"INFO: {len(records)} records from {input_file}")

        name = os.path.splitext(os.path.basename(input_file))[0]
        correlation = calculate_correlation(records, args.label1, args.label2, name)
        summary_outputfile = os.path.join(args.output, f"correlation_{name}.txt")
        print(f"INFO: Writing results for correlation to file {summary_outputfile}")

        with open(summary_outputfile, "w", encoding="utf-8") as outfile:
            outfile.write(correlation)


if __name__ == "__main__":
    main()
