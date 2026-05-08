"""
Model A - Supervised Learning (MEMORY-OPTIMIZED)
Uses SGDClassifier for CPU-based incremental learning without loading entire dataset.
Usage: python model_a_supervised_lite.py
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_LR_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ANSWER_MAP,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
)
from utils import save_model, load_encoder

def load_features_batch(features_path, batch_size=5000):
    """
    Generator to load features in batches without loading entire file into memory.
    
    Args:
        features_path (str): Path to .npz file
        batch_size (int): Number of samples per batch
        
    Yields:
        np.array: Feature batch
    """
    data = np.load(features_path, allow_pickle=True)
    passages = data['passages']
    questions = data['questions']
    options = data['options']
    
    num_samples = passages.shape[0]
    print(f"Total samples: {num_samples}")
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Combine features for this batch
        batch_features = np.hstack([
            passages[start_idx:end_idx],
            questions[start_idx:end_idx],
            options[start_idx:end_idx, 0, :],
            options[start_idx:end_idx, 1, :],
            options[start_idx:end_idx, 2, :],
            options[start_idx:end_idx, 3, :],
        ])
        
        yield batch_features


def load_labels_batch(csv_path, batch_size=5000):
    """
    Generator to load labels in batches.
    
    Args:
        csv_path (str): Path to CSV
        batch_size (int): Number of samples per batch
        
    Yields:
        np.array: Label batch
    """
    df = pd.read_csv(csv_path)
    labels = df["answer"].map(ANSWER_MAP).values
    
    num_samples = len(labels)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        yield labels[start_idx:end_idx]


def train_sgd_classifier(batch_size=5000, n_epochs=1):
    """
    Train SGDClassifier incrementally with batches.
    Memory efficient: never loads entire dataset at once.
    
    Args:
        batch_size (int): Samples per batch
        n_epochs (int): Number of passes through dataset
    """
    print("\n" + "="*60)
    print("Training SGDClassifier (Memory-Optimized)")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print("="*60)
    
    # Initialize model
    model = SGDClassifier(
        loss='log_loss',  # Logistic regression
        max_iter=1,
        warm_start=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Train incrementally
    print("\nTraining...")
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        batch_num = 0
        
        # Load and train on batches
        features_gen = load_features_batch(TRAIN_FEATURES_PATH, batch_size)
        labels_gen = load_labels_batch(TRAIN_CSV_PATH, batch_size)
        
        for batch_features, batch_labels in zip(features_gen, labels_gen):
            model.partial_fit(
                batch_features,
                batch_labels,
                classes=np.array([0, 1, 2, 3])
            )
            batch_num += 1
            print(f"  Batch {batch_num} processed ({len(batch_labels)} samples)")
    
    print("\n✓ Training complete")
    
    # Evaluate on validation set (batched)
    print("\nEvaluating on validation set...")
    val_predictions = []
    val_true = []
    
    features_gen = load_features_batch(VAL_FEATURES_PATH, batch_size)
    labels_gen = load_labels_batch(VAL_CSV_PATH, batch_size)
    
    for batch_features, batch_labels in zip(features_gen, labels_gen):
        batch_preds = model.predict(batch_features)
        val_predictions.extend(batch_preds)
        val_true.extend(batch_labels)
    
    val_predictions = np.array(val_predictions)
    val_true = np.array(val_true)
    
    val_accuracy = accuracy_score(val_true, val_predictions)
    
    # Print results
    print(f"\n{'='*60}")
    print("SGDClassifier (Validation Set)")
    print(f"{'='*60}")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        val_true,
        val_predictions,
        target_names=["A", "B", "C", "D"],
        digits=4
    ))
    print(f"{'='*60}\n")
    
    # Save model
    save_model(model, MODEL_LR_PATH)
    
    return model, val_accuracy


def main():
    print("\n" + "#"*60)
    print("# QuizForge - Model A: Supervised Learning (LITE)")
    print("# CPU-Only, Memory-Optimized Incremental Training")
    print("#"*60)
    
    # Train
    model, val_acc = train_sgd_classifier(batch_size=5000, n_epochs=1)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Model saved to: {MODEL_LR_PATH}")
    print("="*60)
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    main()
