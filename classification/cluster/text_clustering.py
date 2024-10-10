import abc
from abc import ABC
from typing import List


class ClusteredDataPoint:
    def __init__(self, cluster_id: int, cluster_score: float, data_point: dict):
        """
        :param cluster_id: can be any string
        :param cluster_score: How strongly assigned to the cluster. Any number. Optimally normalized to value between 0...1
        :param data_point: The cluster data
        """
        self.cluster_id = cluster_id
        self.cluster_score = cluster_score
        self.data_point = data_point


class Cluster:
    def __init__(self, cluster_id: int, cluster_description: str):
        """
        :param cluster_id: can be any string
        :param cluster_score: How strongly assigned to the cluster. Any number. Optimally normalized to value between 0...1
        :param meta_information: any text information you would like to add to the result
        """
        self.cluster_id = cluster_id
        self.cluster_decription = cluster_description
        self.data_points: ClusteredDataPoint = []
        self.cluster_center = None

    def add_data_point(self, data_point: ClusteredDataPoint):
        self.data_points.append(data_point)

    def __repr__(self):
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
        :return: List of predicted classes (for most classifier, just one class)
        """
        """
        Return an ordered list of Clusters
        """
        return []
