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



td = TfidfVectorizer(max_features=3000)
count_vec = CountVectorizer(max_features=3000)

svm = Pipeline(
    steps=[
        ("TfidfVectorizer", td),
        ("SVC", SVC(kernel="sigmoid", gamma=1, C=0.5, probability=True)),
    ]
)

mnb = Pipeline(
    steps=[("CountVectorizer", count_vec), ("MultinomialNB", MultinomialNB())]
)

DCT = Pipeline(
    steps=[
        ("TfidfVectorizer", td),
        (
            "DecisionTreeClassifier",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=50,
                random_state=42,
                algorithm="SAMME",
            ),
        ),
    ]
)

xgbc = Pipeline(
    steps=[
        ("TfidfVectorizer", td),
        ("XGBClassifier", XGBClassifier(n_estimators=50, random_state=42)),
    ]
)



def model(x_train, y_train):
    votting_system = VotingClassifier(
        estimators=[
            ("SVM", svm),
            ("DTC", DCT),
            ("xgbc", xgbc),
            ("mnb", mnb),
        ],
        voting="soft",
    )
    votting_system.fit(x_train, y_train)
    return votting_system