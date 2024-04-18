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