"""
Multi-Disorder ML Model Training Pipeline
Uses "Silver Labeling" from the DSM-5 Diagnostic Engine to train a multi-class 
text classifier on the Suicide_Detection.csv dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
from diagnostic_engine import DiagnosticAssistant

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def silver_label_data(df, assistant):
    """
    Labels text data using the rule-based Diagnostic Assistant.
    """
    print("Starting Silver Labeling process...")
    labels = []
    
    # Process in batches for progress tracking
    total = len(df)
    for i, text in enumerate(df['text']):
        if (i+1) % 5000 == 0:
            print(f"Processed {i+1}/{total} rows...")
            
        # We only need the score calculation, not the full longitudinal analysis
        results = assistant.analyze([{"text": text, "date": "2000-01-01"}])
        
        # Find the best matching condition
        best_condition = "None/General"
        max_score = 15 # Minimum threshold to consider it a "marker"
        
        for diag in results['diagnostics']:
            if diag['Score'] > max_score:
                max_score = diag['Score']
                best_condition = diag['Condition']
        
        labels.append(best_condition)
    
    return labels

def main():
    print("="*60)
    print("MULTI-DISORDER ML TRAINING PIPELINE")
    print("="*60)

    # 1. Load Data
    data_path = os.path.join(SCRIPT_DIR, 'Suicide_Detection.csv')
    print(f"Loading data from {data_path}...")
    # Sampling 30,000 for faster training in this environment
    df = pd.read_csv(data_path, nrows=30000)
    
    # 2. Silver Labeling
    assistant = DiagnosticAssistant()
    df['labeled_condition'] = silver_label_data(df, assistant)
    
    print("\nLabel Distribution:")
    print(df['labeled_condition'].value_counts())
    
    # 3. Filter and Prepare
    # We want a balanced model, so we keep the "None/General" but maybe not too many
    # However, for now, we'll train on all discovered labels.
    
    X = df['text']
    y = df['labeled_condition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. TF-IDF and Model
    print("\nTraining Multi-Class Model...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save Model
    model_path = os.path.join(SCRIPT_DIR, 'multi_condition_model.pkl')
    vectorizer_path = os.path.join(SCRIPT_DIR, 'multi_tfidf.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n✓ Multi-class model saved to: {model_path}")
    print(f"✓ Vectorizer saved to: {vectorizer_path}")
    print("="*60)

if __name__ == "__main__":
    main()
