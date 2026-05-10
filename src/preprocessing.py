"""
Data preprocessing and splitting for RACE dataset.
Loads the single CSV and creates 80/10/10 train/val/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_and_split_race_dataset(
    input_csv: str,
    output_dir: str = "./data/processed",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> tuple:
    """
    Load RACE dataset and split into train/val/test.
    
    Args:
        input_csv: Path to the input CSV file
        output_dir: Directory to save processed CSVs
        train_size: Proportion for training (default 0.8)
        val_size: Proportion for validation (default 0.1)
        test_size: Proportion for testing (default 0.1)
        random_state: Seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    
    # Load dataset
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Drop index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    
    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    
    # Verify columns
    required_cols = ["id", "article", "question", "A", "B", "C", "D", "answer"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Dataset shuffled. Shape: {df.shape}")
    
    # First split: 80% train, 20% temp (for val/test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state
    )
    
    # Second split: split temp into val (50%) and test (50%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )
    
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} rows")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Load from archive and save to processed
    input_file = "./data/archive/train.csv"
    output_directory = "./data/processed"
    
    if not os.path.exists(input_file):
        # Try alternate path if running from different directory
        input_file = "./archive/train.csv"
    
    train_df, val_df, test_df = load_and_split_race_dataset(
        input_csv=input_file,
        output_dir=output_directory
    )
    
    print("\n-> Data preprocessing complete!")
