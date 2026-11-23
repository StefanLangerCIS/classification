""" Classifier based on hugginface transformers
    Derived from TextClassifier
"""
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import Dataset
from classification.classify.text_classifier import TextClassifier, ClassifierResult
from classification.encode.text_encoder import TextEncoder
from classification.env.env import TMP_DATA_DIR
from utils.app_logging import app_logger

class TransformerClassifier(TextClassifier):
    """
    Classify with a transformer classifier
    """


    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        """
        Initialize the transformer classifier.

        :param model_name: Hugging Face model identifier for tokenizer and model.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

        self.classifier_type = "transformer"
        self.classifier_name = self.classifier_type + "_" + self.model_name

    def name(self) -> str:
        return self.classifier_name

    def info(self) -> Dict[str, Any]:
        return {"type": "transformer", "name": self.classifier_name, "model": self.model_name}

    def classify(self, data: dict) -> List[ClassifierResult]:
        """
        Classify a data point.

        :param data: The data dictionary with field 'text'.
        :return: A list with one classifier result
        """
        predicted_class = ""

        result = ClassifierResult(predicted_class, -1, "")
        return [result]

    def train(self, training_data: List[Dict]) -> None:
        """
        Train the classifier
        :param training_data: List of training data points with fields 'text' and 'label'.
        :return: Nothing
        """
        ds = Dataset.from_list(training_data)
        label_names = sorted(set(ds["label"]))
        label2id = {name: i for i, name in enumerate(label_names)}
        ds = ds.map(lambda x: {"label": label2id[x["label"]]})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(label_names)
        )
        ds_split = ds.train_test_split(test_size=0.2)
        training_dataset = ds_split['train']
        validation_dataset = ds_split['test']
        if len(validation_dataset) < 3:
            app_logger.error("Data set too small")
        training_dataset = training_dataset.map(self._tokenize, batched=True)
        validation_dataset = validation_dataset.map(self._tokenize, batched=True)
        training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir=TMP_DATA_DIR,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch"
        )


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer
        )

        trainer.train()


    def _tokenize(self, datapoints: dict) -> dict:
        """Tokenize a batch of text data points."""
        return self.tokenizer(datapoints['text'], padding='max_length', truncation=True)


    def _save_model_information(self) -> None:
        """
        Print detailed information about the model
        :return: None
        """
        pass
        # if self.classifier_type == "DecisionTreeClassifier":
        #    self.print_decision_tree()
        # else:
        #    pass

