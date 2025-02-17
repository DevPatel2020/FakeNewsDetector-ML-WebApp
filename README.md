# Fake News Detection Webapp

## Description

This advanced Streamlit application detects fake news using machine learning and provides comprehensive model insights. Key features include:

- Multiple vectorizer options (TF-IDF, Bag of Words, Hashing Vectorizer)
- Various classifiers (SVM, Naive Bayes, Logistic Regression, Random Forest)
- Interactive visualization dashboard
- Model diagnostics (confusion matrix, precision-recall curves)
- Feature importance analysis
- Real-time performance metrics

## Installation

1. Clone the repository
2. Install requirements:

```bash
pip install -r requirements.txt
```

## Enhanced Modules

### Module 3: Expanded Model Selection
- **Vectorizers**: 
  - TF-IDF (Unigram/Bigram/Trigram)
  - Bag of Words
  - Hashing Vectorizer
- **Classifiers**:
  - Linear SVM
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Passive Aggressive Classifier

### Module 5: Advanced Analytics
- Real-time model metrics (Accuracy, F1 Score, Recall)
- Interactive visualizations:
  - Confusion Matrix
  - Precision-Recall Curves
  - Calibration Plots
  - Feature Importance Charts
  - Word Frequency Analysis

### Module 6: Visualization Dashboard
- Class distribution analysis
- Word clouds for text visualization
- Side-by-side comparison of real/fake news word frequencies
- Model calibration curves

## Dataset
- Uses the "fake_or_real_news.csv" dataset
- Automatic train/test split (80/20)
- Binary classification (0 = Real, 1 = Fake)

## Usage

1. Start the app:

```bash
streamlit run main.py --client.showErrorDetails=false
```

2. Interface features:
   - Left sidebar for model configuration
   - Main panel for text input and results
   - Interactive visualization tabs
   - Real-time performance metrics

3. Analysis workflow:
   - Paste news text in the input area
   - Select vectorizer/classifier combination
   - Click "Analyze Article" for instant prediction
   - Explore model diagnostics and visualizations

## Key Enhancements
- Added 5 new visualization types
- Implemented 2 additional classifiers
- Introduced bigram/trigram support
- Created interactive model diagnostics
- Added feature importance analysis
- Improved UI with custom styling
- Added performance metric tracking

