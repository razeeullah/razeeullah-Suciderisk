"""
Multi-Disorder ML Model Training Pipeline
Version: 2.1.0 (Viva Edition)

Uses "Silver Labeling" from the rule-based DSM-5 Diagnostic Engine 
to train multi-class text classifiers for mental health conditions.
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Modules from the local project
from diagnostic_engine import DiagnosticAssistant

# Suppress minor scikit-learn warnings for cleaner viva demonstration
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'Suicide_Detection.csv')
SAMPLE_SIZE = 5000  # Size for silver labeling demonstration

def silver_label_data(df, assistant):
    """
    Labels text data using the rule-based Diagnostic Assistant.
    This creates 'Silver Labels' for supervised learning.
    """
    print(f"\n[1/3] Starting Silver Labeling on {len(df)} rows...")
    labels = []
    
    for i, text in enumerate(df['text']):
        if (i+1) % 1000 == 0:
            print(f"   > Processed {i+1} rows...")
            
        # Analyze text to find clinical markers
        results = assistant.analyze([{"text": text, "date": "viva-2024-01-01"}])
        
        # Determine the primary condition based on the highest score
        best_condition = "None/General"
        max_score = 15 # Minimum threshold for a specific diagnosis
        
        for diag in results['diagnostics']:
            if diag['Score'] > max_score:
                max_score = diag['Score']
                best_condition = diag['Condition']
        
        labels.append(best_condition)
    
    print("✓ Labeling complete.")
    return labels

def main():
    print("="*60)
    print("MENTAL HEALTH AI - MULTI-CONDITION TRAINING")
    print("="*60)

    # 1. Load Data
    print(f"Loading raw data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)
    except Exception as e:
        print(f"✖ Error: {e}"); return

    # 2. Silver Labeling
    assistant = DiagnosticAssistant()
    df['labeled_condition'] = silver_label_data(df, assistant)
    
    print("\nClinical Distribution found in sample:")
    print(df['labeled_condition'].value_counts())
    
    # 3. Data Preparation
    X = df['text']
    y = df['labeled_condition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Extraction & Multi-Algorithm Training
    print("\n[2/3] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Define models for comparison
    models_to_train = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'linear_regression': RidgeClassifier(class_weight='balanced'),
        'svm': SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
    }
    
    print("\n[3/3] Training and evaluating algorithms...")
    for name, model in models_to_train.items():
        print(f"\n--- Training {name.upper()} ---")
        model.fit(X_train_tfidf, y_train)
        
        # Test performance
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # Save each model variation
        model_path = os.path.join(SCRIPT_DIR, f'multi_{name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"✓ Saved model: {model_path}")

    # Save the common vectorizer
    vectorizer_path = os.path.join(SCRIPT_DIR, 'multi_tfidf.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✓ Saved global vectorizer: {vectorizer_path}")
    
    print("\n" + "="*60)
    print("MULTI-CONDITION PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
