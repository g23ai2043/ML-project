import streamlit as st
import re
import nltk
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Function to load the machine learning model
def load_model():
    lr = joblib.load('models/logistic_regression_model.pkl')
    return lr

# Load the sentiment model
model = load_model()

# Data Cleaning Function 
def data_cleaning(text):
    mention_pattern = r'@\S+|#\S+'  # Pattern to match mentions (words starting with @ and #)
    url_pattern = r"https?:(?:www\.)?\S+"  # Pattern to match URLs (starting with http or https)
    non_alphanumeric_pattern = r"[^A-Za-z0-9\s]+"  # Pattern to match non-alphanumeric characters (including punctuation)

    # Lowercasing 
    lower_text = str.lower(text)
    
    # Removing all unnecessary data 
    clean_text = re.sub(mention_pattern + '|' + url_pattern + '|' + non_alphanumeric_pattern, ' ', lower_text)
    
    return clean_text   

def data_prep_txt(text):
    ls=[]
    ls.append(text)
    df=pd.DataFrame(ls)
    df.columns =['text']

    stop_words = set(stopwords.words('english'))
# Performing Data Cleaning
    df['text'] = df['text'].apply(data_cleaning)
    df['tokenized text'] = df['text'].apply(lambda document: word_tokenize(document.strip()))
    df['clean_tokens'] = df['tokenized text'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    lemmatizer = WordNetLemmatizer()

# Applying Lemmatization
    df['lemmatized text'] = df['clean_tokens'].apply(lambda tokens: ' '.join([lemmatizer.lemmatize(token, pos='v') for token in tokens]))

# Creating new features
    df['no_of_charcters'] = df['lemmatized text'].apply(len)
    df['no_of_words'] = df['lemmatized text'].apply(lambda document: word_tokenize(document)).apply(len)
    return df['lemmatized text']

def data_prep_series(df):
    

    stop_words = set(stopwords.words('english'))
# Performing Data Cleaning
    df['text'] = df['text'].apply(data_cleaning)
    df['tokenized text'] = df['text'].apply(lambda document: word_tokenize(document.strip()))
    df['clean_tokens'] = df['tokenized text'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
    lemmatizer = WordNetLemmatizer()

# Applying Lemmatization
    df['lemmatized text'] = df['clean_tokens'].apply(lambda tokens: ' '.join([lemmatizer.lemmatize(token, pos='v') for token in tokens]))

# Creating new features
    df['no_of_charcters'] = df['lemmatized text'].apply(len)
    df['no_of_words'] = df['lemmatized text'].apply(lambda document: word_tokenize(document)).apply(len)
    return df['lemmatized text']


# Streamlit App UI
st.title('Sentiment Analysis App')

# Text input sentiment analysis
st.header("Single Text Input for Sentiment Analysis")
st.write('Enter some text to analyze its sentiment:')
input_text = st.text_area('Text Input', '')

# Predict sentiment for single text
if st.button('Analyze Sentiment'):
    if input_text.strip() == '':
        st.write("Please enter some text!")
    else:
        prediction = model.predict(data_prep_txt([input_text]))[0]
        sentiment = prediction
        
        st.subheader("Sentiment Prediction:")
        st.write(f"The sentiment of the entered text is **{sentiment}**.")

# File upload for bulk sentiment analysis
st.header("CSV File Upload for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Ensure the CSV has the expected column (e.g., 'text')
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            # Predict sentiment for each text in the 'text' column
            df['Sentiment'] = model.predict(data_prep_series(df))
           
            # Display the dataframe
            st.subheader('Sentiment Analysis Results')
            st.dataframe(df, height=400)  # Display CSV with scroll

            # Convert DataFrame to CSV for download
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_result = convert_df_to_csv(df)

            # Download button for results
            st.download_button(
                label="Download CSV with Predictions",
                data=csv_result,
                file_name="sentiment_predictions.csv",
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Error processing the file: {e}")
