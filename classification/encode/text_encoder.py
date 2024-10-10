from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle

from classification.encode.text_embedder import TextEmbedder
from utils.app_logging import app_logger


class TextEncoder(ABC):
    """
    Abstract base class for preprocessing data for a classifier.
    """
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def create_matrix(self, data: pd.DataFrame, train: bool = False) -> numpy.ndarray:
        pass


class TextEncoderSparse(TextEncoder):
    """
    Preprocessing of the data - transform text into numerical features.
    """
    def __init__(self, lsa=False):
        self.lsa = lsa
        # Features for sparse and LSA variants
        # The minimum number of document a word has to occur in to be considered
        self.min_df = 3
        # The maximum fraction of documents a word might occur in to be considered
        self.max_df = 0.8
        # Minimum length of word sequence for sparse matrix
        self.words_ngram_min = 1
        # Maximum length of word sequence for sparse matrix
        self.words_ngram_max = 3

        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.svd_transformer = None

    def name(self) -> str:
        nm = "sparse"
        if self.lsa:
            nm += "_lsa"
        return nm

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "type": "sparse",
            "lsa": str(self.lsa),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "words_ngram_min": self.words_ngram_min,
            "words_ngram_max": self.words_ngram_max,
        }

    def create_matrix(self, data: pd.DataFrame, train: bool = False) -> numpy.ndarray:
        """
        Create the data matrix used for classifier training and classification.

        :param data: A pandas data frame with the data
        :param train: If creating a tf-idf matrix - train the matrix.
        :return: The created matrix.
        """
        if train:
            self._train_tfidf_matrix(data)
        elif self.count_vectorizer is None:
            message = "CountVectorizer has not been trained. Call create matrix first with 'train=True'"
            app_logger.error(message)
            raise RuntimeError(message)
        matrix = self._create_tfidf_matrix(data)
        return matrix

    def _train_tfidf_matrix(self, data_train: pd.DataFrame) -> None:
        """
        Based on a pandas data frame, build the TF-IDF table
        :param data_train: Data frame containing the text in column 'text'
        :return: Nothing
        """
        app_logger.info("Creating TF-IDF Matrix")
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                min_df=self.min_df, max_df=self.max_df, ngram_range=(self.words_ngram_min, self.words_ngram_max)
            )
        matrix_train_counts = self.count_vectorizer.fit_transform(data_train.text)
        if self.tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(
                matrix_train_counts
            )
        app_logger.info("TF-IDF matrix training completed")
        if self.lsa:
            app_logger.info("Reducing matrix with LSA")
            matrix_train = self.tfidf_transformer.transform(matrix_train_counts)
            self.svd_transformer = TruncatedSVD(n_components=200, random_state=0)
            self.svd_transformer.fit(matrix_train)
            app_logger.info("LSA matrix reduction completed")
        return None

    def _create_tfidf_matrix(self, data_to_transform: pd.DataFrame) -> numpy.ndarray:
        matrix_counts = self.count_vectorizer.transform(data_to_transform.text)
        matrix = self.tfidf_transformer.transform(matrix_counts)
        if self.lsa:
            matrix = self.svd_transformer.transform(matrix)
        else:
            matrix = matrix.toarray()
        return matrix


class TextEncoderDense(TextEncoder):
    """
    Preprocessing of the data - transform text into numerical features - embedding.
    """
    def __init__(self):
        self.sentence_transformer = None
        # Model selected for good speed and ok-ish performance
        # see https://www.sbert.net/docs/pretrained_models.html
        # Multilingual
        # Multilingual: "paraphrase-multilingual-MiniLM-L12-v2"
        # Monolingual: "all-MiniLM-L12-v2"
        self.model = "paraphrase-multilingual-MiniLM-L12-v2"

    def name(self) -> str:
        return f"dense_{self.model}"

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "type": "dense",
            "model": self.model,
        }

    def create_matrix(self, data: pd.DataFrame, train: bool = False) -> numpy.ndarray:
        """
        Create a dense matrix based on some deep learning model
    
        :param data: The data to create the dense matrix from
        :param train: In the dense preprocessor, this parameter is ignored.
        :return: The dense matrix.
        """ ""
        if self.sentence_transformer is None:
            self.sentence_transformer = TextEmbedder(
                self.model, cache_file=None
            )

        encoded = self.sentence_transformer.encode(data.text)
        return encoded


def create_data_table(data_records: List) -> pd.DataFrame:
    """
    Create a shuffled data frame.

    :param data_records: The data records to create the data table from
    :return: The panda data frame
    """
    data_records = shuffle(data_records)
    data_table = pd.DataFrame(data_records)
    data_table.fillna(0, inplace=True)
    return data_table
