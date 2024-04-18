<br />
<p align="center">

  <h3 align="center">ML Project for Email spam Detection</h3>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Description](#description)
- [Dataset](#Dataset)
- [Packages and frameworks i used](#packages-and-frameworks-i-used)
- [Installing packages](#installing-packages)
- [Deplyment and run the application](#deplyment-the-application)

## Description

This repository contains code and resources for detecting spam emails using machine learning techniques. Email spam, also known as unsolicited bulk email (UBE), is a prevalent issue in the digital world, leading to inbox clutter and potential security risks. By analyzing the content and metadata of emails, we aim to develop models that can accurately classify emails as spam or non-spam.

## Dataset

The dataset used in this project from Kaggle, to download it use this [link](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)

## Packages and frameworks i used

* [pandas](https://pandas.pydata.org/docs/) for datasets
* [sklearn](https://scikit-learn.org/stable/index.html) for machine learning models and data cleaning
* [nltk](https://www.nltk.org/api/nltk.html) for text preprocessing 
* [streamlit](https://docs.streamlit.io/) for deployment
* [seaborn](https://seaborn.pydata.org/) and [matplotlib](https://matplotlib.org/) for data analysis and visualization and EDA

## Installing packages

#### open Terminal in VS code, and write following commands
        pip install pandas
        pip install sklearn
        pip install streamlit
        pip install seaborn
        pip install matplotlib
        pip install nltk

## Deplyment and run the application

#### open Terminal in VS code, and write following commands
        cd "deployment with streamlit"
        streamlit run app.py