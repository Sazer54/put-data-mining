# Twitter Bot Classifier

A web application for classifying tweets as BOT or HUMAN using a selected classifier (Naive Bayes, SVM, Random Forest).

## Requirements

- Python 3.8+
- Installed libraries:  
  `streamlit`, `nltk`, `scikit-learn`, `pickle`

You can install the required libraries with:
```
pip install streamlit nltk scikit-learn
```

## Preparation

1. Make sure the `models/` directory contains the following files (you can download the testing versions from: https://drive.google.com/drive/folders/1Gq-zlNypqq9DlsNXHx8MOhn_uuZ2bxmS?usp=sharing:
   - `nb_classifier.pkl`
   - `svm_classifier.pkl`
   - `rf_classifier.pkl`
   - `vectorizer.pkl`

2. If you are running the application for the first time, download the NLTK resources (the code does this automatically, but an internet connection is required).

## Running the application

In the terminal, navigate to the directory with `main.py` and run:

```
streamlit run main.py
```

The application will open in your browser.

## Usage

1. Select a classifier from the dropdown list.
2. Paste or type the tweet text into the text field.
3. Click the **Predict** button.
4. Read the classification result: BOT or HUMAN.

---

**Note:**  
If you want to train your own models on your own dataset, please visit /notebooks

# put-data-mining
The project associated with the Data Mining course at the Poznań University of Technology

Collaborators:
 - Szymon Skowroński
 - Samuel Levi Paszyński
