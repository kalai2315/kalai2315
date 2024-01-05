## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Dependencies](#dependencies)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

# Spam Message Classification

## Introduction

### 1.1 Project Overview

This project focuses on building a machine learning model for classifying messages as either spam or non-spam. It employs natural language processing and machine learning techniques to automatically identify and filter out spam messages.

### 1.2 Objective

The primary goal is to develop an accurate and efficient classification model that can help users distinguish between spam and legitimate messages in their communication channels.

### 1.3 Key Features

- **Machine Learning Model:** Utilizes  **RandomForestClassifier**, **Support Vector Machines (SVM)** algorithm for effective message classification.
- **Data Preprocessing:**  TF-IDF Vectorization
- **Performance Metrics:** Evaluates the model using classification_report, accuracy_score, confusion_matrix to ensure robustness.


## Data

### 2.1 ## Dataset Description

The dataset used in this Spam Message Classification project is structured as a tabular dataset with the following columns:

- **Label:** The target variable indicating whether a message is spam or non-spam (ham).
- **Message:** The actual content of the text message.
- **Length:** The length of the message in terms of the number of characters.
- **Punct:** The count of punctuation marks in the message.

### 2.2 ## Dataset Overview

- **Size:** The dataset contains a total of 5572 messages.
- **Classes:**
  - **Spam:** 4825
  - **Non-Spam (Ham):** 747

### Features

1. **Label:**
   - The 'Label' column contains binary values representing the classification of each message. '1' may denote spam, and '0' may denote non-spam.

2. **Message:**
   - The 'Message' column contains the raw text content of each message.

3. **Length:**
   - The 'Length' column represents the number of characters in each message.

4. **Punct:**
   - The 'Punct' column indicates the count of punctuation marks in each message.


### 2.3 Data Preprocessing

Before training the machine learning model, the text data undergoes preprocessing, including the use of the `TfidfVectorizer` for feature extraction.

### TF-IDF Vectorization

The `TfidfVectorizer` is employed to convert a collection of raw text messages into a matrix of TF-IDF features. This technique transforms the text data into numerical vectors, capturing the importance of words within each document.


#### Example code snippet for TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

#### Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

#### Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(messages)

#### tfidf_matrix now contains the TF-IDF representation of the text data

## 3. Model Training

### 3.1 Algorithm Selection

The machine learning model in this project utilizes the **Support Vector Machines (SVM)** algorithm for effective message classification. SVM is a powerful and versatile algorithm known for its ability to handle high-dimensional data and nonlinear relationships.

### 3.2 Understanding SVM

Support Vector Machines work by finding the hyperplane that best separates different classes in the feature space. In the case of spam message classification, the algorithm learns to distinguish between spam and non-spam messages based on the features extracted from the text data.

### 3.3 SVM Parameters

The SVM implementation may involve tuning various parameters for optimal performance. Key parameters include the choice of kernel (linear, polynomial, or radial basis function), regularization parameters (C), and others. Adjust these parameters based on the characteristics of your dataset.


#### Example code snippet for SVM training with scikit-learn
from sklearn.svm import SVC

svm = Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", SVC(C = 100, gamma='auto'))])

svm.fit(X_train, y_train)


### 4.  Dependencies

### 4.1 Required Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

### 4.2 Installation

To run the Spam Message Classification project, ensure you have the required Python libraries installed. If you don't have them installed yet, use the following commands:

pip install numpy

pip install pandas

pip install matplotlib

pip install scikit-learn

pip install nltk

python -m nltk.downloader all

### 5. Evaluation

The performance of the Spam Message Classification model is assessed using various metrics to ensure its effectiveness in distinguishing between spam and non-spam messages.

#### Model Evaluation Metrics

**Accuracy:**
  - Accuracy measures the overall correctness of the model predictions.

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = classifier.predict(X_test)

y_pred = svm.predict(X_test)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))




