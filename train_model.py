"""
Machine Learning Model Training for Suicide Detection
Version: 2.1.0 (Viva Edition)

This script implements a multi-model training pipeline for binary text classification.
It compares Linear Regression, Logistic Regression, and Support Vector Machines (SVM).
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Suppress minor scikit-learn warnings for a cleaner output during demonstration
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = 'Suicide_Detection.csv'
MAX_FEATURES = 5000  # Number of words to consider in TF-IDF
SAMPLE_SIZE = 20000  # Reduced for demonstration performance

class SuicideDetectionModel:
    """
    Mental Health ML Pipeline: Handles data loading, text preprocessing, 
    feature extraction (TF-IDF), and multi-algorithm training.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_name = None
        self.training_stats = {}
        
    def load_data(self, filepath=DATA_FILE, sample_size=SAMPLE_SIZE):
        """
        Load the suicide detection dataset.
        """
        if not os.path.isabs(filepath):
            filepath = os.path.join(SCRIPT_DIR, filepath)
        
        print(f"\n[1/5] Loading data from {filepath}...")
        
        try:
            if sample_size:
                df = pd.read_csv(filepath, nrows=sample_size)
            else:
                df = pd.read_csv(filepath)
            
            print(f"✓ Loaded {len(df):,} rows.")
            print("\n--- Distribution of Classes ---")
            print(df['class'].value_counts())
            return df
        except Exception as e:
            print(f"✖ Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Standardizes input text by removing distractions like punctuation,
        extra whitespace, and URLs.
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, df):
        """
        Splits data into training and testing sets.
        """
        print("\n[2/5] Preprocessing and splitting data...")
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        df = df[df['cleaned_text'].str.len() > 0]
        df['label'] = (df['class'] == 'suicide').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'],
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        print(f"✓ Training samples: {len(X_train):,}")
        print(f"✓ Testing samples:  {len(X_test):,}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='logistic_regression'):
        """
        Trains the selected machine learning algorithm.
        """
        self.model_name = model_type
        print(f"\n[3/5] Starting {model_type} training...")
        
        # 1. Feature Extraction
        if self.vectorizer is None:
            print(f"Building TF-IDF model (max_features={MAX_FEATURES})...")
            self.vectorizer = TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, 2),
                stop_words='english'
            )
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
        else:
            X_train_tfidf = self.vectorizer.transform(X_train)
            
        # 2. Algorithm Initialization
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 3. Fit
        start = datetime.now()
        self.model.fit(X_train_tfidf, y_train)
        duration = (datetime.now() - start).total_seconds()
        
        print(f"✓ Training completed in {duration:.2f} seconds.")
        self.training_stats['training_time'] = duration
    
    def evaluate(self, X_test, y_test):
        """
        Measures model performance on unseen data.
        """
        print(f"\n[4/5] Evaluating {self.model_name}...")
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Raw predictions
        y_raw = self.model.predict(X_test_tfidf)
        
        # Post-process for Linear Regression (thresholding)
        if self.model_name == 'linear_regression':
            y_pred = (y_raw >= 0.5).astype(int)
        else:
            y_pred = y_raw
            
        acc = accuracy_score(y_test, y_pred)
        print(f"--- {self.model_name.upper()} RESULTS ---")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Suicide Risk']))
        
        self.training_stats['accuracy'] = acc
    
    def save(self):
        """
        Serialization of models for use in the app.
        """
        print(f"\n[5/5] Saving {self.model_name}...")
        m_filename = f"suicide_{self.model_name}_model.pkl"
        v_filename = "tfidf_vectorizer.pkl"
        
        joblib.dump(self.model, os.path.join(SCRIPT_DIR, m_filename))
        joblib.dump(self.vectorizer, os.path.join(SCRIPT_DIR, v_filename))
        print(f"✓ Saved to {m_filename}")

def main():
    print("="*60)
    print("MENTAL HEALTH AI - SUICIDE DETECTION TRAINING")
    print("="*60)
    
    pipeline = SuicideDetectionModel()
    df = pipeline.load_data()
    
    if df is not None:
        X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
        
        # Systematic training for viva demonstration
        algorithms = ['linear_regression', 'logistic_regression', 'svm']
        
        for algo in algorithms:
            pipeline.train_model(X_train, y_train, model_type=algo)
            pipeline.evaluate(X_test, y_test)
            pipeline.save()
            
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
