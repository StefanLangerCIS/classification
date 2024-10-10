"""
Run clustering for various algorithms
"""
import argparse
import json
import os
import time
import random

from classification.cluster.sklearn_clustering import SklearnClustering

# Data directory. Follow structure in _default_data_dir or
# set the CLS_SRCH_DATA_DIR environment directory
_this_dir = os.path.abspath(os.path.dirname(__file__))
_default_data_dir = os.path.abspath(os.path.join(_this_dir, "../../../", "data", "news", "clustering"))
DATA_DIR = os.getenv("CLS_SRCH_DATA_DIR", _default_data_dir)


def configure_filters(filter_args_string: str) -> dict:
    """
    Configure the filters from the runtime argument string
    :param filter_args_string: The args string
    :return: Filters as dictionary
    """
    filters = {}
    if filter_args_string is not None and ":" in filter_args_string:
        filter_strings = filter_args_string.split(",")
        for filter_string in filter_strings:
            (field, value) = filter_string.split(":")
            filters[field] = value
    return filters


def filter_match(record: dict, filters: dict) -> bool:
    """
    Check whether filters match for data record
    :param record: record to check
    :param filters: filter(s) to match
    :return: True is all filters pass, else False
    """
    # Each filter must match
    for field in filters:
        if field not in record:
            return False
        if record[field] != filters[field]:
            return False
    # All filters passed (or no filters were present)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or several text classifiers"
    )

    # All available clustering algorithm types
    algorithm_types = SklearnClustering.supported_algos

    parser.add_argument(
        "--input",
        default=os.path.join(
            DATA_DIR, "clustering_data_full.jsonl"
        ),
        help="Data for clustering",
    )

    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "results"),
        help="Folder where to write the clustering results",
    )

    parser.add_argument(
        "--algorithm",
        choices=algorithm_types + ["all"],
        # default="KMeans",
        default="all",
        help="The clustering algorithm to use. If 'all' iterate through all available algorithms",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default="5",
        help="The number of clusters to create",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default="10000",
        help="The number of samples to use. use 0 for all",
    )

    parser.add_argument(
        "--text_labels",
        default="headline,short_description",
        help="Label/field in the json data contains the text to cluster",
    )

    parser.add_argument(
        "--filter",
        # default="lang:de",
        default="",
        help="Filter for choosing the data (e.g. language)",
    )

    parser.add_argument("--verbose", action="store_true", help="Provide verbose output")

    args = parser.parse_args()

    # Run all clustering algorithms
    if args.algorithm == "all":
        algorithms = algorithm_types
    else:
        algorithms = [args.algorithm]

    print(f"INFO: Running algorithm(s) {algorithms}")

    filters = configure_filters(args.filter)

    # read data
    records = []
    with open(args.input, encoding="utf-8") as input_fp:
        for line in input_fp:
            record = json.loads(line)
            if filter_match(record, filters):
                records.append(record)

    print(f"INFO: {len(records)} records from {args.input} after filtering with {filters}")

    if args.n_samples > 0:
        records = random.sample(records, args.n_samples)

    text_labels = args.text_labels.split(",")

    # Iterate over the clustering algorithms

    for algorithm in algorithms:
        clustering = SklearnClustering(algorithm, args.n_clusters)
        clustering.verbose = args.verbose
        print(f"INFO: Running clustering with algorithm {algorithm} for {len(records)} records")
        clustering_time = time.time()
        clusters = clustering.cluster(records, text_labels)
        clustering_time = int(time.time() - clustering_time)
        print(
            f"INFO: Clustering completed for algorithm {algorithm} in {clustering_time} seconds"
        )

        summary_outputfile = os.path.join(args.output, f"results_{algorithm}.txt")
        print(
            f"INFO: Writing results for algorithm {algorithm} to file {summary_outputfile}"
        )

        with open(summary_outputfile, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write(f"Clustering algorithm: {algorithm}\n")
            outfile.write("\n#Counts:\n")
            outfile.write(f"Number of data_records: {len(records)}\n")
            for cluster in clusters:
                outfile.write("###\n")
                outfile.write(
                    f"Cluster: {cluster.cluster_id}, n_records: {len(cluster.data_points)}, desc {cluster.cluster_decription}\n"
                )
                for record in cluster.data_points[0:30]:
                    outfile.write(
                        "{0}\n".format(json.dumps(record, ensure_ascii=False))
                    )
                outfile.write("###\n")
            outfile.write("\n#Performance:\n")
            outfile.write(f"Seconds used for clustering: {clustering_time}\n")

        data_outputfile = os.path.join(args.output, f"clusters_{algorithm}.jsonl")
        with open(data_outputfile, "w", encoding='utf-8') as outfile:
            for cluster in clusters:
                for data_point in cluster.data_points:
                    data_point["cluster_id"] = str(cluster.cluster_id)
                    outfile.write(json.dumps(data_point, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
