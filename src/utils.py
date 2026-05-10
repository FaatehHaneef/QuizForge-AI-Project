"""
Utility functions for QuizForge project
Shared helpers for loading data, saving models, and inference
"""

import numpy as np
import pandas as pd
import joblib
from config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    ANSWER_MAP,
    ANSWER_MAP_REVERSE,
    ENCODER_PATH,
)


def load_features(features_path):
    """
    Load feature matrices from .npz file.
    Contains: passages, questions, options, and optionally labels.
    
    Args:
        features_path (str): Path to .npz file
        
    Returns:
        dict: Dictionary with 'passages', 'questions', 'options' arrays
    """
    print(f"Loading features from {features_path}...")
    data = np.load(features_path, allow_pickle=True)
    
    # Extract features
    passages = data['passages']
    questions = data['questions']
    options = data['options']  # Shape: (num_samples, 4, 5000) for 4 options
    
    print(f"-> Loaded features:")
    print(f"  - Passages: {passages.shape}")
    print(f"  - Questions: {questions.shape}")
    print(f"  - Options: {options.shape}")
    
    # Combine all features: passages + questions + all 4 options
    # Shape: (num_samples, 5000 + 5000 + 5000*4) = (num_samples, 30000)
    num_samples = passages.shape[0]
    combined_features = np.hstack([
        passages,  # (num_samples, 5000)
        questions,  # (num_samples, 5000)
        options[:, 0, :],  # Option A: (num_samples, 5000)
        options[:, 1, :],  # Option B: (num_samples, 5000)
        options[:, 2, :],  # Option C: (num_samples, 5000)
        options[:, 3, :],  # Option D: (num_samples, 5000)
    ])  # Result: (num_samples, 30000)
    
    print(f"  - Combined features: {combined_features.shape}")
    
    return combined_features


def load_labels_from_csv(csv_path):
    """
    Load labels (answer column) from CSV and convert to numeric.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        np.array: Numeric labels (0=A, 1=B, 2=C, 3=D)
    """
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    labels = df["answer"].map(ANSWER_MAP).values
    print(f"-> Loaded {len(labels)} labels")
    print(f"  Label distribution: {np.bincount(labels)}")
    return labels


def load_train_data():
    """Load train features and labels."""
    train_features = load_features(TRAIN_FEATURES_PATH)
    train_labels = load_labels_from_csv(TRAIN_CSV_PATH)
    return train_features, train_labels


def load_val_data():
    """Load validation features and labels."""
    val_features = load_features(VAL_FEATURES_PATH)
    val_labels = load_labels_from_csv(VAL_CSV_PATH)
    return val_features, val_labels


def load_test_data():
    """Load test features and labels."""
    test_features = load_features(TEST_FEATURES_PATH)
    test_labels = load_labels_from_csv(TEST_CSV_PATH)
    return test_features, test_labels


def save_model(model, save_path):
    """
    Save model to disk using joblib.
    
    Args:
        model: Trained sklearn model
        save_path (str): Path to save model
    """
    print(f"Saving model to {save_path}...")
    joblib.dump(model, save_path)
    print(f"-> Model saved successfully")


def load_model(model_path):
    """
    Load model from disk using joblib.
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        Trained model
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print(f"-> Model loaded successfully")
    return model


def load_encoder():
    """
    Load feature encoder (OneHotEncoder).
    
    Returns:
        sklearn encoder object
    """
    print(f"Loading encoder from {ENCODER_PATH}...")
    encoder = joblib.load(ENCODER_PATH)
    print(f"-> Encoder loaded successfully")
    return encoder


def evaluate_model(model, features, labels, model_name="Model"):
    """
    Evaluate model accuracy on features and labels.
    
    Args:
        model: Trained sklearn model
        features: Feature matrix (sparse or dense)
        labels: True labels
        model_name (str): Name for logging
        
    Returns:
        float: Accuracy score
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        labels,
        predictions,
        target_names=["A", "B", "C", "D"],
        digits=4
    ))
    print(f"{'='*50}\n")
    
    return accuracy


def predict(model, features):
    """
    Make predictions on features.
    
    Args:
        model: Trained sklearn model
        features: Feature matrix
        
    Returns:
        np.array: Predicted labels (0/1/2/3)
    """
    return model.predict(features)


def predict_proba(model, features):
    """
    Get prediction probabilities.
    
    Args:
        model: Trained sklearn model (must support predict_proba)
        features: Feature matrix
        
    Returns:
        np.array: Probability matrix
    """
    return model.predict_proba(features)
