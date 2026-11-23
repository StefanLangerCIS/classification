import abc
from abc import ABC
from typing import List


class ClusteredDataPoint:
    """A single data point assigned to a cluster."""

    def __init__(self, cluster_id: int, cluster_score: float, data_point: dict):
        """
        :param cluster_id: Numeric cluster identifier.
        :param cluster_score: Cluster membership score, ideally normalized to 0..1.
        :param data_point: The original data record.
        """
        self.cluster_id = cluster_id
        self.cluster_score = cluster_score
        self.data_point = data_point


class Cluster:
    """A cluster containing data points and descriptive metadata."""

    def __init__(self, cluster_id: int, cluster_description: str):
        """
        :param cluster_id: Numeric cluster identifier.
        :param cluster_description: Human-readable description of the cluster.
        """
        self.cluster_id = cluster_id
        self.cluster_decription = cluster_description
        self.data_points: List[ClusteredDataPoint] = []
        self.cluster_center: list | None = None

    def add_data_point(self, data_point: ClusteredDataPoint) -> None:
        """Add a data point to this cluster."""
        self.data_points.append(data_point)

    def __repr__(self) -> str:
        return f"{self.__class__} object. cluster_id: {self.cluster_id}, cluster_description: {self.cluster_decription}, number_of_datapoints: {len(self.data_points)}"


class TextClustering(ABC):
    """
    Abstract base class for clustering
    """

    @abc.abstractmethod
    def cluster(self, data: List[dict], text_label: str) -> List[Cluster]:
        """
        Cluster a list of data_records
        :param data: List of dictionary (parsed json) - the data_records to cluster
        :param text_label: the dictionary/json field in the data_records which contains the text
        :return: List of clusters.
        """
        return []
