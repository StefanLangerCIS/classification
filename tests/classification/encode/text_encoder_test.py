import os
import unittest
from typing import Tuple

import numpy

from classification.encode import text_encoder
from classification.encode.text_encoder import TextEncoderDense, create_data_table, TextEncoderSparse


class TestTextEncoder(unittest.TestCase):
    def test_text_encoder_dense(self):
        text_encoder = TextEncoderDense()
        dt = create_data_table([{"text": "To be or not to be?", "label": "ws"}])
        matrix = text_encoder.create_matrix(dt)
        self.assertTrue(isinstance(matrix, numpy.ndarray))

    def test_text_encoder_sparse(self):
        text_encoder = TextEncoderSparse()
        data_list = []
        for text in ["To be or not to be?", "We are such stuff as dreams are made of.", ""]*10:
            data_list.append({"text": text, "label": "ws"})
        dt = create_data_table(data_list)
        matrix = text_encoder.create_matrix(dt, train=True)
        self.assertTrue(isinstance(matrix, numpy.ndarray))


if __name__ == "__main__":
    unittest.main()
