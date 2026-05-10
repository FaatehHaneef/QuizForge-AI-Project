"""
Feature Engineering Pipeline for RACE Dataset
- Manual text preprocessing
- One-Hot Encoding implementation (custom & sklearn)
- Feature matrix creation for passages, questions, answer options
- Save/load for fast model training
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Dict, List


class TextPreprocessor:
    """Manual text preprocessing pipeline"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        - Lowercase
        - Remove URLs
        - Remove special characters (keep only alphanumeric and spaces)
        - Remove extra whitespace
        """
        # Handle NaN/None values
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        text = text.strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple tokenization: split by spaces and remove punctuation from tokens
        """
        # Remove punctuation while splitting
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        # Remove empty tokens and single chars
        tokens = [t for t in tokens if len(t) > 1]
        return tokens


class OneHotEncoder:
    """
    Manual One-Hot Encoder implementation
    Creates binary vectors where each dimension = 1 word in vocabulary
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        """
        Args:
            max_vocab_size: Max vocabulary size (most frequent words)
            min_freq: Minimum frequency for a word to be included
        """
        self.vocab = {}  # word -> index
        self.vocab_list = []  # index -> word
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.preprocessor = TextPreprocessor()
    
    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts
        """
        word_freq = defaultdict(int)
        
        # Count word frequencies
        for text in texts:
            cleaned = self.preprocessor.clean_text(text)
            tokens = self.preprocessor.tokenize(cleaned)
            for token in tokens:
                word_freq[token] += 1
        
        # Filter by min frequency and sort by frequency
        vocab_words = [word for word, freq in word_freq.items() 
                      if freq >= self.min_freq]
        vocab_words = sorted(vocab_words, 
                            key=lambda w: word_freq[w], 
                            reverse=True)[:self.max_vocab_size]
        
        # Build vocab dictionaries
        self.vocab_list = vocab_words
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        
        print(f"-> Vocabulary built: {len(self.vocab)} words")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text as One-Hot vector
        Returns binary vector of shape (vocab_size,)
        """
        cleaned = self.preprocessor.clean_text(text)
        tokens = set(self.preprocessor.tokenize(cleaned))  # Use set for unique words
        
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] = 1.0
        
        return vector
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of texts
        Returns matrix of shape (n_texts, vocab_size)
        """
        vectors = np.array([self.encode(text) for text in texts], dtype=np.float32)
        return vectors


class FeatureEngineer:
    """
    Main feature engineering pipeline
    Creates feature matrices for Model A (Answer Verification)
    """
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.encoder = OneHotEncoder(max_vocab_size=vocab_size, min_freq=min_freq)
        self.preprocessor = TextPreprocessor()
    
    def fit_encoder(self, df: pd.DataFrame) -> None:
        """
        Fit encoder on all text data (passages + questions + options)
        """
        print("Building vocabulary from all texts...")
        
        # Combine all text
        all_texts = list(df['article']) + list(df['question'])
        for col in ['A', 'B', 'C', 'D']:
            all_texts.extend(list(df[col]))
        
        self.encoder.fit(all_texts)
    
    def create_features(self, df: pd.DataFrame, encode_options: bool = True) -> Dict[str, np.ndarray]:
        """
        Create feature matrices for the dataset
        
        For Model A, each sample becomes:
        [passage_vector] + [question_vector] + [option_A_vector] + ... + [option_D_vector]
        
        Args:
            df: DataFrame with columns: article, question, A, B, C, D, answer
            encode_options: Whether to encode individual options or combined
        
        Returns:
            Dictionary with:
            - 'passages': (n_samples, vocab_size)
            - 'questions': (n_samples, vocab_size)
            - 'options': (n_samples, 4, vocab_size) - for each option
            - 'labels': (n_samples,) - correct answer index (0-3 for A-B-C-D)
            - 'answer_text': (n_samples,) - original answer letter
        """
        print(f"Creating features for {len(df)} samples...")
        
        n_samples = len(df)
        vocab_size = len(self.encoder.vocab)
        
        # Initialize arrays
        passages = np.zeros((n_samples, vocab_size), dtype=np.float32)
        questions = np.zeros((n_samples, vocab_size), dtype=np.float32)
        options = np.zeros((n_samples, 4, vocab_size), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int32)
        
        # Encode each field
        for idx in range(n_samples):
            passages[idx] = self.encoder.encode(df.iloc[idx]['article'])
            questions[idx] = self.encoder.encode(df.iloc[idx]['question'])
            
            for opt_idx, opt_letter in enumerate(['A', 'B', 'C', 'D']):
                options[idx, opt_idx] = self.encoder.encode(df.iloc[idx][opt_letter])
            
            # Convert answer letter to index (A=0, B=1, C=2, D=3)
            answer_letter = df.iloc[idx]['answer']
            labels[idx] = ord(answer_letter) - ord('A')
        
        print(f"-> Features created: passages={passages.shape}, questions={questions.shape}, "
              f"options={options.shape}")
        
        return {
            'passages': passages,
            'questions': questions,
            'options': options,
            'labels': labels,
            'answer_text': df['answer'].values
        }
    
    def save_features(self, features: Dict, output_path: str) -> None:
        """Save features to compressed numpy format (faster than pickle)"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_path_npz = output_path.replace('.pkl', '.npz')
        
        # Save as compressed numpy file
        np.savez_compressed(
            output_path_npz,
            passages=features['passages'],
            questions=features['questions'],
            options=features['options'],
            labels=features['labels']
        )
        print(f"-> Saved features to {output_path_npz}")
    
    def load_features(self, path: str) -> Dict:
        """Load features from compressed numpy file"""
        path_npz = path.replace('.pkl', '.npz')
        data = np.load(path_npz)
        features = {
            'passages': data['passages'],
            'questions': data['questions'],
            'options': data['options'],
            'labels': data['labels']
        }
        print(f"-> Loaded features from {path_npz}")
        return features
    
    def save_encoder(self, output_path: str) -> None:
        """Save encoder (vocab) to pickle file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        print(f"-> Saved encoder to {output_path}")
    
    def load_encoder(self, path: str) -> None:
        """Load encoder from pickle file"""
        with open(path, 'rb') as f:
            self.encoder = pickle.load(f)
        print(f" -> Loaded encoder from {path}")


def main():
    """
    Main pipeline: load data → fit encoder → create features → save
    """
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Paths
    data_dir = "./data/processed"
    output_dir = "./data/features"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n1. Loading datasets...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize feature engineer
    print("\n2. Initializing feature engineer...")
    fe = FeatureEngineer(vocab_size=5000, min_freq=3)
    
    # Fit encoder on TRAINING DATA ONLY (to avoid data leakage)
    print("\n3. Fitting encoder on training data...")
    fe.fit_encoder(train_df)
    
    # Create features for each split
    print("\n4. Creating features...")
    print("\n   Train split:")
    train_features = fe.create_features(train_df)
    
    print("\n   Validation split:")
    val_features = fe.create_features(val_df)
    
    print("\n   Test split:")
    test_features = fe.create_features(test_df)
    
    # Save features
    print("\n5. Saving features and encoder...")
    fe.save_features(train_features, f"{output_dir}/train_features.pkl")
    fe.save_features(val_features, f"{output_dir}/val_features.pkl")
    fe.save_features(test_features, f"{output_dir}/test_features.pkl")
    fe.save_encoder(f"{output_dir}/encoder.pkl")
    
    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"\nVocabulary size: {len(fe.encoder.vocab)}")
    print(f"Feature vector size: {len(fe.encoder.vocab)}")
    print(f"\nOutput files:")
    print(f"  - {output_dir}/train_features.pkl")
    print(f"  - {output_dir}/val_features.pkl")
    print(f"  - {output_dir}/test_features.pkl")
    print(f"  - {output_dir}/encoder.pkl")
    print("\nReady for Model A training!")


if __name__ == "__main__":
    main()
