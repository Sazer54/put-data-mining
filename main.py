import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from pickle import load


# Pobranie zasobów WordNet
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Załadowanie modeli
with open('models/nb_classifier.pkl', 'rb') as f:
    nb_classifier = load(f)
with open('models/svm_classifier.pkl', 'rb') as f:
    svm_classifier = load(f)
with open('models/rf_classifier.pkl', 'rb') as f:
    rf_classifier = load(f)
with open('models/vectorizer.pkl','rb') as f:
    vectorizer = load(f)

#Funkcja preprocessingu tekstu
def filter_text(text):
    tokens = word_tokenize(text)
    lemmas = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in lemmas if word.lower() not in stop_words]
    return ' '.join(filtered)

#funkcje predykcji
def predict_spam_nb(text):
    filtered_text = filter_text(text)
    X = vectorizer.transform([filtered_text])
    return nb_classifier.predict(X)[0]

def predict_spam_svm(text):
    filtered_text = filter_text(text)
    X = vectorizer.transform([filtered_text])
    return svm_classifier.predict(X)[0]

def predict_spam_rf(text):
    filtered_text = filter_text(text)
    X = vectorizer.transform([filtered_text])
    return rf_classifier.predict(X)[0]

# Aplikacja webowa
st.title('TWITTER BOT CLASSIFIER')
classifier_option = st.selectbox('Choose a classifier', ['Naive Bayes', 'SVM', 'Random Forest'])
input_text = st.text_area('Enter the text of the tweet')

if st.button('Predict'):
    filtered_text = filter_text(input_text)
    vectorized_text = vectorizer.transform([filtered_text])
    if classifier_option == 'Naive Bayes':
        prediction = nb_classifier.predict(vectorized_text)[0]
    elif classifier_option == 'SVM':
        prediction = svm_classifier.predict(vectorized_text)[0]
    elif classifier_option == 'Random Forest':
        prediction = rf_classifier.predict(vectorized_text)[0]
    else:
        pass
    result = 'BOT' if prediction == 1 else 'HUMAN'
    st.write(f'The tweet is probably: {result}')

