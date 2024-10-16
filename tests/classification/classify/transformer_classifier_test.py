import json
import os
import unittest
from typing import Tuple

from datasets import Dataset
from pyarrow.dataset import dataset
from transformers import AutoTokenizer

from classification.classify.run_classifier import run_classifier, ClassificationResultData
from classification.classify.text_classifier import get_data_records_from_file
from classification.classify.transformer_classifier import TransformerClassifier
from tests.classification.classify.run_classifier_test import get_test_files

def tokenize_function(datapoints):
    model_name = "distilbert-base-uncased"  # Use any desired model checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(datapoints['text'], padding='max_length', truncation=True)


class TestRunClassifier(unittest.TestCase):
    def test_dataset(self):
        train_file, _ , _ = get_test_files()

        ds = Dataset.from_json(train_file)
        ds = ds.rename_column('author', 'label')
        train_test_split = ds.train_test_split(test_size=0.2)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        self.assertTrue(eval_dataset.shape)
        self.assertTrue(train_dataset.shape)

    def test_train(self):
        classifier = TransformerClassifier()
        train_file, _, _ = get_test_files()
        training_data = get_data_records_from_file(train_file, ["text"], "author")
        classifier.train(training_data)

    def test_download(self):
        # If this fails, try to run several times from console first...
        from transformers import AutoModel
        model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.assertTrue(model)



if __name__ == "__main__":
    unittest.main()
