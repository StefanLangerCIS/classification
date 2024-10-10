"""
This file defines the environment for the project
"""
import os

DATA_DIR = r"C:\ProjectData\Uni\classif_srch\data\letters"
CLASSIFIER_DATA_DIR = os.path.join(DATA_DIR, "classification")
CLASSIFIER_TRAINING_FILE = os.path.join(CLASSIFIER_DATA_DIR, "classifier_data_train.jsonl")
CLASSIFIER_EVALUATION_FILE = os.path.join(CLASSIFIER_DATA_DIR, "classifier_data_eval.jsonl")

CLASSIFIER_OUTPUT_DIR = os.path.join(CLASSIFIER_DATA_DIR, "results")
