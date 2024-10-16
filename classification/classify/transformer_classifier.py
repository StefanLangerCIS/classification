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

"""
TBD: this is only a skeleton.

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# 1. Load the dataset (we'll use the 'emotion' dataset as an example)
dataset = load_dataset("emotion")

# 2. Load a pre-trained model and tokenizer



# 3. Preprocess the text data


# 4. Fine-tune the model




# 5. Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)


# 6. Run the model to predict the class of new texts
def get_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    return dataset["train"].features["label"].int2str(probs.argmax().item())


# Example usage
print(get_prediction("I feel great today!"))

"""

class TransformerClassifier(TextClassifier):
    """
    Classify with a transformer classifier
    """


    def __init__(self, model_name = "distilbert-base-multilingual-cased"):
        """
        Initialize the classifier.


        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=6)

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
        ds_split = ds.train_test_split(test_size=0.2)
        training_dataset = ds_split['train']
        validation_dataset = ds_split['test']
        if len(validation_dataset) < 3:
            app_logger.error("Data set too small")
        # training_dataset = training_dataset.map(self._tokenize, batched=True)
        # validation_dataset = validation_dataset.map(self._tokenize, batched=True)
        # training_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        # validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir=TMP_DATA_DIR,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )


        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()


    def _tokenize(self, datapoints):
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

    def _print_decision_tree(self):
        """
        Print decision tree rules
        :return:
        """
        rules_text = export_text(self.sklearn_classifier, max_depth=100)
        # Vocabulary for replacement in the data which contains
        # feature numbers only
        if isinstance(self.preprocessor, TextEncoderSparse):
            vocab = self.preprocessor.count_vectorizer.vocabulary_
        else:
            app_logger.warning("Cannot print decision tree.")
            return
        vocabulary = dict((feature, word) for word, feature in vocab.items())
        rules = rules_text.split("\n")
        lines = []
        for rule in rules:
            if "feature_" in rule:
                word_id_str = re.sub(".*feature_([0-9]+).*", r"\1", rule)
                word_id = int(word_id_str)
                if word_id in vocabulary:
                    word = vocabulary[word_id]
                else:
                    word = "UNK"
                rule = rule.replace(f"feature_{word_id_str}", word)
                lines.append(rule)
            else:
                lines.append(rule)

        with open(
                os.path.join(self.model_folder_path, "decision_rules.txt"),
                "w",
                encoding="utf-8",
        ) as out:
            for line in lines:
                out.write(line + "\n")

"""
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

# Example list of dictionaries
data = [
    {'text': 'I love this movie!', 'author': 1},
    {'text': 'This movie was terrible.', 'author': 0},
    # More examples...
]

# Convert list of dictionaries to Dataset
dataset = Dataset.from_list(data)

# Rename the column 'author' to 'label'
dataset = dataset.rename_column('author', 'label')

# Optionally, split the data into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load a pre-trained tokenizer
model_name = "distilbert-base-uncased"  # Use any desired model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Ensure 'num_labels' matches your task

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
)

# Train the model
trainer.train()

"""