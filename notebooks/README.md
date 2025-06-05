# Twitter Bot Classifier – Backend

This Jupyter Notebook (`Backend.ipynb`) is responsible for preparing data, training, evaluating, and saving machine learning models for classifying Twitter users as bots or humans.

## Features

- Loads and preprocesses tweet data and user labels.
- Balances the dataset between bots and humans.
- Tokenizes and cleans tweet text (BPE tokenization, stopword removal, lemmatization).
- Vectorizes text using TF-IDF.
- Trains and evaluates three classifiers:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Prints accuracy, precision, recall, F1 score, classification report, and confusion matrix for each model.
- Saves trained models and the vectorizer as `.pkl` files for later use in the web application.

## Requirements

- Python 3.8+
- Jupyter Notebook
- Required libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `scikit-learn`
  - `tokenizers`
  - `pickle`

Install dependencies with:
```
pip install pandas numpy nltk scikit-learn tokenizers
```

## Usage

1. Place your data files (`tweet_subset_15.json` and `label.csv`) in the working directory.
2. Open `Backend.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells to preprocess data, train models, and save the results.
4. The trained models and vectorizer will be saved as `.pkl` files (e.g., `nb_classifier.pkl`, `svm_classifier.pkl`, `rf_classifier.pkl`, `vectorizer.pkl`).

## Output

- Trained model files in the working directory.
- Evaluation metrics printed for each classifier.
- Preprocessed data ready for use in the Streamlit frontend (`main.py`).

---

**Note:**  
Make sure to use the same vectorizer and models in your frontend application for consistent predictions.
