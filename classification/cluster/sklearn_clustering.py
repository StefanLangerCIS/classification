""" Clustering algorithms based on sklearn
    They all have a similar interface, so the can be wrapped in one class
    Generic interface defined in TextClustering
"""
from collections import Counter
from typing import List

import numpy as np
import pandas
from sklearn.cluster import OPTICS, AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestCentroid
from sklearn.utils import shuffle
from text_clustering import Cluster, TextClustering


class SklearnClustering(TextClustering):
    """
    Clustering with sklearn cluster algorithms
    """

    supported_algos = [
        "KMeans",
        "AgglomerativeClustering_ward",
        "AgglomerativeClustering_single",
        "AgglomerativeClustering_complete",
        "AgglomerativeClustering_average",
        "OPTICS",
    ]

    def __init__(self, algorithm: str, n_clusters: int = 5, verbose=False):
        """
        Initialize the classifier
        :param algorithm: The name of the clustering algorithm
        :param n_c
        lusters: Number of clusters. Ignored for density based algorithms
        :param verbose: Print more...
        """
        # Store the file path of the training data
        self.data = None
        self.verbose = verbose

        if algorithm == "KMeans":
            self.sklearn_clustering = KMeans(verbose=verbose, n_clusters=n_clusters)
        elif algorithm.startswith("AgglomerativeClustering"):
            algo, linkage_method = algorithm.split("_")
            self.sklearn_clustering = AgglomerativeClustering(
                linkage=linkage_method, n_clusters=n_clusters
            )
        elif algorithm == "OPTICS":
            self.sklearn_clustering = OPTICS()
        else:
            raise Exception(
                "Unsupported clustering type {0}. Use one of {1}".format(
                    algorithm, self.supported_algos
                )
            )

        self.algorithm = algorithm
        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.text_labels = None
        self.data_table = None
        self.matrix_tfidf = None

    def cluster(self, data_records: List[dict], text_labels: List[str]) -> List[Cluster]:
        """
        Cluster
        :param data_records: List of dicts (data records)
        :param text_labels: Json fields which contain the text
        :return: The list of created clusters
        """
        """
        Train the algorithm with the data from the knowledge graph
        """
        self.text_labels = text_labels
        self.data_table = self._create_data_table(data_records, text_labels)
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                min_df=5, max_df=0.3, ngram_range=(1, 1)
            )
        matrix_counts = self.count_vectorizer.fit_transform(self.data_table.text)
        if self.tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(matrix_counts)
        self.matrix_tfidf = self.tfidf_transformer.transform(matrix_counts).toarray()
        self.sklearn_clustering.fit_predict(self.matrix_tfidf)
        return self._build_result()

    # ********************************
    # Creation  of data table from data data_records
    # ********************************
    def _create_data_table(
        self, data_records: List[dict], text_labels: List[str]
    ) -> pandas.DataFrame:
        """
        :param data_records: the records to create the data table for
        :param text_labels: The labels of the fields which contains the text
        :return: The data frame
        """
        self.data_records = {}
        data_points = []
        record_number = 0
        for record in data_records:
            record_number += 1
            # Store the original data records. We want to add them to the cluster info later
            self.data_records[record_number] = record
            datapoint = {}
            datapoint["text"] = ". ".join([record.get(label, "") for label in text_labels])
            datapoint["id"] = record_number
            data_points.append(datapoint)

        return self._create_data_frame(data_points)

    def _create_data_frame(self, data_points: List[dict]) -> pandas.DataFrame:
        """
        Create the data table
        :param data_points: The list of data points as a list of dictionaries:
        :return: The data frame
        """
        # shuffle for good measure
        data_points = shuffle(data_points)
        data_table = pandas.DataFrame(data_points)
        data_table = data_table.fillna(0)
        return data_table

    # ********************************
    # Result building
    # ********************************
    def _build_result(self) -> List[Cluster]:
        """
        Create the result objects (cluster list)
        :param data_table: The data table which was clustered
        :return: A list of clusters
        """
        clusters = {}
        cluster_centers = self._compute_cluster_centers()
        for cluster_id in self.sklearn_clustering.labels_:
            if cluster_id in clusters:
                continue
            cluster = Cluster(cluster_id, str(cluster_id))
            cluster.cluster_center = cluster_centers[cluster_id]
            clusters[cluster_id] = cluster
        for i in range(0, len(self.data_table)):
            record_id = self.data_table.iloc[i]["id"]
            record = self.data_records[record_id]
            cluster_id = self.sklearn_clustering.labels_[i]
            clusters[cluster_id].add_data_point(record)
        for cluster in clusters.values():
            self._create_cluster_terms_centroid(cluster)
        return clusters.values()

    def _create_cluster_terms_count(self, cluster: Cluster) -> None:
        """
        Get the significant terms for the cluster and add to description
        using the count of terms
        :param cluster: The cluster to add the description to
        :return:
        """
        top_terms = Counter()
        for record in cluster.data_points:
            text_matrix = self.count_vectorizer.transform([record[self.text_label]])
            feature_names = self.count_vectorizer.get_feature_names()
            matrix_tf_idf = self.tfidf_transformer.transform(text_matrix)
            for tf in matrix_tf_idf:
                sorted_features = np.argsort(tf.data)[: -(50 + 1) : -1]
                for feature in sorted_features:
                    feature_name = feature_names[tf.indices[feature]]
                    top_terms[feature_name] += 1
        cluster.cluster_decription = str(top_terms.most_common(10))

    def _create_cluster_terms_centroid(self, cluster: Cluster) -> None:
        """
        Get the significant terms for the cluster and add to description
        using the cluster centers
        :param cluster: The cluster to add the diescription to
        :return:
        """
        feature_names = self.count_vectorizer.get_feature_names_out()
        terms_with_tfidf = []
        for i in range(0, len(cluster.cluster_center)):
            value = cluster.cluster_center[i]
            term = feature_names[i]
            terms_with_tfidf.append((term, value))

        terms_with_tfidf.sort(key=lambda x: x[1], reverse=True)
        cluster.cluster_decription = str(terms_with_tfidf[0:10])

    def _compute_cluster_centers(self):
        """
        Computer the cluster center
        :param labels: The labels and the clustered data
        :param data_table: The data table to use
        :return: The cluster centers
        """

        if self.algorithm == "KMeans":
            return self.sklearn_clustering.cluster_centers_
        else:
            # Train classifier for the computed clusters and use the centroid
            centroid_classifier = NearestCentroid()
            centroid_classifier.fit(self.matrix_tfidf, self.sklearn_clustering.labels_)
            return centroid_classifier.centroids_
