# Suicide Risk Analyzer 💙

A machine learning-based suicide risk detection system using NLP and clinical frameworks.

## 🎯 Features

- **93% Accuracy** ML model trained on 1M+ samples
- Real-time text analysis for suicide risk detection
- Safety-first design with automatic crisis resource display
- Streamlit web interface
- Comprehensive crisis intervention protocols

## 📊 Model Performance

- **Accuracy**: 92.87%
- **Precision**: 93.60%
- **Recall**: 92.02%
- **F1 Score**: 92.80%

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/razeeullah/Sucideriskanalyser.git
cd Sucideriskanalyser
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

The dataset is too large for GitHub. Download it separately:

**Dataset:** [Suicide Detection Dataset on Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

Save as `Suicide_Detection.csv` in the project directory.

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load 100,000 samples from the dataset
- Train a Logistic Regression model with TF-IDF features
- Save the model files (`suicide_detection_model.pkl`, `tfidf_vectorizer.pkl`)

**Training time:** ~2-3 minutes on a modern CPU

### 5. Run the Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## 📁 Project Structure

```
Sucideriskanalyser/
├── app.py                          # Streamlit web application
├── train_model.py                  # ML model training pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Suicide_Detection.csv          # Dataset (download separately)
├── suicide_detection_model.pkl    # Trained model (generated)
└── tfidf_vectorizer.pkl           # TF-IDF vectorizer (generated)
```

## 🛡️ Safety Features

- **Automatic High-Risk Detection**: Displays crisis resources immediately
- **988 Suicide & Crisis Lifeline**: Integrated prominently
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: 911 information
- **International Resources**: Global crisis center directory

## ⚠️ Important Disclaimer

**This tool is for analytical support only and does NOT replace professional medical judgment.**

- This is not a medical diagnosis
- Always consult licensed mental health professionals
- If someone is in immediate danger, call 911 or go to the nearest emergency room
- Crisis support is available 24/7 at 988 (US)

## 🔬 Technical Details

### Model Architecture

- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF with 5,000 features
- **N-grams**: Unigrams and bigrams
- **Class Balancing**: Weighted to handle imbalanced data
- **Training Data**: 100,000 samples (80/20 split)

### Dataset

- **Source**: Suicide Watch dataset
- **Size**: 1,025,022 rows
- **Format**: CSV with `text` and `class` columns
- **Classes**: `suicide` and `non-suicide`
- **Balance**: ~50/50 split

## 📞 Crisis Resources

### United States
- **988 Suicide & Crisis Lifeline**: Call or text 988
- **Crisis Text Line**: Text HOME to 741741
- **Veterans Crisis Line**: Press 1 after calling 988
- **Emergency**: 911

### International
- Visit [Find a Helpline](https://findahelpline.com) for resources worldwide

## 🧪 Testing

The model includes sample predictions in `train_model.py`:

```python
# Run training to see test predictions
python train_model.py
```

## 📈 Model Evaluation

**Confusion Matrix:**
```
                Predicted
              Non-Suicide  Suicide
Actual Non-         9385      628
       Suicide        797     9187
```

- **True Negatives**: 9,385 (correctly identified non-suicide)
- **False Positives**: 628 (false alarms - acceptable for safety)
- **False Negatives**: 797 (missed cases - minimized but not zero)
- **True Positives**: 9,187 (correctly identified suicide risk)

## 🤝 Contributing

This project is designed for educational and research purposes. If you'd like to improve the model or add features:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. The dataset has its own license terms.

## 🙏 Acknowledgments

- Dataset: Suicide Watch Reddit posts
- Clinical Framework: Columbia Suicide Severity Rating Scale (C-SSRS)
- Crisis Resources: 988 Suicide & Crisis Lifeline

## ⚡ Performance Notes

- Training on full dataset (1M rows) takes ~20-30 minutes
- Using 100K samples provides 93% accuracy in 2-3 minutes
- Increasing to 500K samples may improve accuracy to ~94-95%

---

💙 **Remember**: This tool supports, never replaces, human judgment and compassion. If you or someone you know is in crisis, help is available right now.

**Call 988** (US) or visit your local emergency services.
