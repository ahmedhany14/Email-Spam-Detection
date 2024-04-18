# reading datasets
import pandas as pd

# algebra
import numpy as np

# strings
import string


# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# for the most used words
from wordcloud import WordCloud


# text preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# splitting the data
from sklearn.model_selection import train_test_split

# models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# pipelines
from sklearn.pipeline import Pipeline

# for evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix

# for votting system model
from sklearn.ensemble import VotingClassifier

import streamlit as st
import text_preprocessing as tp
import machine_learning_pipelines as mlp

emails = pd.read_csv(
    r"/home/ahmed/Ai/Data science and Ml projects/Email-Spam-Detection/datasets/SPAM text message 20170820 - Data.csv"
)
emails = emails.drop_duplicates()
def update(cat):
    if cat == "spam":
        return 1
    elif cat == "ham":
        return 0
    return cat

def fixed_text(text):
    
    # defining an object from the text_preprocessing class
    process = tp.text_preprocessing()

    # apply the process in the text
    text = process.convert_text_to_lower(text)
    words = process.convert_text_to_words(text)
    words = process.remove_special_characters(words)
    words = process.remove_stop_words(words)
    words = process.remove_punctuation(words)
    words = process.stem_the_words(words)
    ret = " ".join(words)
    return ret

emails["Category"] = emails["Category"].apply(update)
emails["Preprcessed_Text"] = emails["Message"].apply(fixed_text)

X = emails["Preprcessed_Text"]
Y = emails["Category"]


model = mlp.model(X, Y)

st.header("Email Spam Classifier")

text = st.text_input("Enter your email")


if text not in [None, ""]:

    st.write(
        "### Email you entered:"
    )
    st.write(text)
    
    text = fixed_text(text)
    result = model.predict([text])
    if result == 1:
        st.write("### This is a spam email")
    else:
        st.write("### This is a ham email")