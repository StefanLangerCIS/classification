""" Classifier based on hugginface transformers
    Derived from TextClassifier
"""

"""
TBD: this is only a skeleton.

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# 1. Load the dataset (we'll use the 'emotion' dataset as an example)
dataset = load_dataset("emotion")

# 2. Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)


# 3. Preprocess the text data
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()

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