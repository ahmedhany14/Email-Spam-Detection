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


class text_preprocessing:
    def __init__(self) -> None:
        pass

    def convert_text_to_lower(self, text=str):
        text = text.lower()
        return text

    def convert_text_to_words(self, text=str):
        words = word_tokenize(text)
        return words

    def remove_special_characters(self, words):

        fixed_words = []

        for w in words:
            if w.isalnum():
                fixed_words.append(w)

        return fixed_words

    def remove_stop_words(self, words):

        fixed_words = []

        stop_words = set(stopwords.words("english"))

        for w in words:
            if not stop_words.__contains__(w):
                fixed_words.append(w)

        return fixed_words

    def remove_punctuation(self, words):

        fixed_words = []
        punctuation = set(string.punctuation)

        for w in words:

            if not punctuation.__contains__(w):
                fixed_words.append(w)

        return fixed_words

    def stem_the_words(self, words):
        fixed_words = []

        ps = PorterStemmer()

        for w in words:
            stem_w = ps.stem(w)
            fixed_words.append(stem_w)

        return fixed_words