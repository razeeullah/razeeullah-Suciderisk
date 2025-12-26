"""
Machine Learning Model Training for Suicide Detection

Trains classification models on the Suicide_Detection.csv dataset
and integrates with the clinical sentiment analyzer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import joblib
import re
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SuicideDetectionModel:
    """
    ML model for suicide risk detection from text.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_name = None
        self.training_stats = {}
        
    def load_data(self, filepath='Suicide_Detection.csv', sample_size=None):
        """
        Load and preprocess the dataset.
        
        Args:
            filepath: Path to CSV file (can be relative to script or absolute)
            sample_size: Optional number of rows to sample (for faster training)
            
        Returns:
            DataFrame with loaded data
        """
        # If filepath is not absolute, make it relative to script directory
        if not os.path.isabs(filepath):
            filepath = os.path.join(SCRIPT_DIR, filepath)
        
        print(f"Loading data from {filepath}...")
        
        try:
            # Load dataset
            if sample_size:
                print(f"Sampling {sample_size} rows...")
                df = pd.read_csv(filepath, nrows=sample_size)
            else:
                df = pd.read_csv(filepath)
            
            print(f"Loaded {len(df):,} rows")
            
            # Check for class distribution
            print("\nClass distribution:")
            print(df['class'].value_counts())
            print(f"\nClass percentages:")
            print(df['class'].value_counts(normalize=True) * 100)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep punctuation that might be meaningful
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with 'text' and 'class' columns
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\nPreprocessing text...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Encode labels (suicide=1, non-suicide=0)
        df['label'] = (df['class'] == 'suicide').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'],
            df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='logistic_regression'):
        """
        Train the classification model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            model_type: 'logistic_regression' or 'random_forest'
        """
        print(f"\n{'='*80}")
        print(f"Training {model_type} model...")
        print(f"{'='*80}")
        
        # Create TF-IDF vectorizer
        print("\nCreating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_name = model_type
        
        print(f"\nTraining model...")
        start_time = datetime.now()
        
        self.model.fit(X_train_tfidf, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.training_stats['training_time'] = training_time
        self.training_stats['num_features'] = X_train_tfidf.shape[1]
        self.training_stats['training_samples'] = len(X_train)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*80}")
        print("MODEL EVALUATION")
        print(f"{'='*80}")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide']))
        
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n                Predicted")
        print(f"              Non-Suicide  Suicide")
        print(f"Actual Non-   {cm[0][0]:>10}  {cm[0][1]:>7}")
        print(f"       Suicide {cm[1][0]:>10}  {cm[1][1]:>7}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        self.training_stats.update(metrics)
        
        return metrics
    
    def predict(self, text):
        """
        Predict suicide risk for given text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (prediction, probability)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess
        cleaned = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return prediction, probability
    
    def save_model(self, model_path='suicide_detection_model.pkl', 
                   vectorizer_path='tfidf_vectorizer.pkl',
                   stats_path='training_stats.pkl'):
        """
        Save trained model and vectorizer.
        
        Args:
            model_path: Path to save model (relative to script directory)
            vectorizer_path: Path to save vectorizer
            stats_path: Path to save training statistics
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save to script directory
        model_path = os.path.join(SCRIPT_DIR, model_path)
        vectorizer_path = os.path.join(SCRIPT_DIR, vectorizer_path)
        stats_path = os.path.join(SCRIPT_DIR, stats_path)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.training_stats, stats_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Vectorizer saved to: {vectorizer_path}")
        print(f"✓ Statistics saved to: {stats_path}")
    
    def load_model(self, model_path='suicide_detection_model.pkl',
                   vectorizer_path='tfidf_vectorizer.pkl',
                   stats_path='training_stats.pkl'):
        """
        Load pre-trained model and vectorizer.
        
        Args:
            model_path: Path to model file
            vectorizer_path: Path to vectorizer file
            stats_path: Path to statistics file
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        try:
            self.training_stats = joblib.load(stats_path)
        except:
            self.training_stats = {}
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Vectorizer loaded from: {vectorizer_path}")


def main():
    """
    Main training pipeline.
    """
    print("="*80)
    print("SUICIDE DETECTION MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Initialize model
    detector = SuicideDetectionModel()
    
    # Load data (use sample for faster training, set to None for full dataset)
    # For 1M rows, sample 100k for reasonable training time
    df = detector.load_data('Suicide_Detection.csv', sample_size=100000)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = detector.prepare_data(df)
    
    # Train model (Logistic Regression recommended for text classification)
    detector.train_model(X_train, y_train, model_type='logistic_regression')
    
    # Evaluate model
    metrics = detector.evaluate_model(X_test, y_test)
    
    # Save model
    detector.save_model()
    
    # Test predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    test_texts = [
        "I'm feeling great today! Life is wonderful.",
        "I can't take this anymore. I want to end it all.",
        "Sometimes I feel sad but I know it will pass.",
        "I have the pills ready. Tonight is the night."
    ]
    
    for text in test_texts:
        prediction, probability = detector.predict(text)
        risk_level = "SUICIDE RISK" if prediction == 1 else "No Risk"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        print(f"\nText: {text[:60]}...")
        print(f"Prediction: {risk_level}")
        print(f"Confidence: {confidence:.2%}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved and ready for use in the application.")


if __name__ == "__main__":
    main()
