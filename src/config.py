"""
Configuration module for QuizForge project.
Centralized paths, constants, and hyperparameters.
"""

import os

# Directory paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FEATURES_DIR, exist_ok=True)

# Feature engineering constants
VOCAB_SIZE = 5000
ENCODING_TYPE = "one_hot"  # or "tfidf"

# Feature file paths
ENCODER_PATH = os.path.join(DATA_FEATURES_DIR, "encoder.pkl")
TRAIN_FEATURES_PATH = os.path.join(DATA_FEATURES_DIR, "train_features.npz")
VAL_FEATURES_PATH = os.path.join(DATA_FEATURES_DIR, "val_features.npz")
TEST_FEATURES_PATH = os.path.join(DATA_FEATURES_DIR, "test_features.npz")

# CSV paths
TRAIN_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "train.csv")
VAL_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "test.csv")

# Model paths
MODEL_LR_PATH = os.path.join(MODELS_DIR, "model_a_supervised_lr.pkl")
MODEL_SVM_PATH = os.path.join(MODELS_DIR, "model_a_supervised_svm.pkl")
MODEL_KMEANS_PATH = os.path.join(MODELS_DIR, "model_a_unsupervised_kmeans.pkl")
MODEL_ENSEMBLE_PATH = os.path.join(MODELS_DIR, "model_a_ensemble.pkl")
MODEL_B_PATH = os.path.join(MODELS_DIR, "model_b.pkl")

# Answer mapping
ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
ANSWER_MAP_REVERSE = {0: "A", 1: "B", 2: "C", 3: "D"}

# Hyperparameters - Logistic Regression
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42
LR_C = 1.0

# Hyperparameters - SVM
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_RANDOM_STATE = 42

# Hyperparameters - K-Means
KMEANS_N_CLUSTERS = 4
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

# Training / Evaluation
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
