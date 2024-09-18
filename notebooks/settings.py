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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
tqdm.pandas()
nltk.download('punkt_tab')


warnings.filterwarnings('ignore')

###Data Cleansing
# All Patterns
mention_pattern = r'@\S+|#\S+'  # Pattern to match mentions (words starting with @ and #)
url_pattern = r"https?:(?:www\.)?\S+"  # Pattern to match URLs (starting with http or https)
non_alphanumeric_pattern = r"[^A-Za-z0-9\s]+"  # Pattern to match non-alphanumeric characters (including punctuation)


# Data Cleaning Function 
def data_cleaning(text):
    # Lowercasing 
    lower_text = str.lower(text)
    
    # Removing all unnecessary data 
    clean_text = re.sub(mention_pattern + '|' + url_pattern + '|' + non_alphanumeric_pattern, ' ', lower_text)
    
    return clean_text

def data_prep(text):
    ls=[]
    ls.append(text)
    df=pd.DataFrame(ls)
    df.columns =['text']

    stop_words = set(stopwords.words('english'))
# Performing Data Cleaning
    df['text'] = df['text'].apply(data_cleaning)
    df['tokenized text'] = df['text'].progress_apply(lambda document: word_tokenize(document.strip()))
    df['clean_tokens'] = df['tokenized text'].progress_apply(lambda tokens: [token for token in tokens if token not in stop_words])
    lemmatizer = WordNetLemmatizer()

# Applying Lemmatization
    df['lemmatized text'] = df['clean_tokens'].progress_apply(lambda tokens: ' '.join([lemmatizer.lemmatize(token, pos='v') for token in tokens]))

# Creating new features
    df['no_of_charcters'] = df['lemmatized text'].progress_apply(len)
    df['no_of_words'] = df['lemmatized text'].progress_apply(lambda document: word_tokenize(document)).apply(len)
    return df['lemmatized text']