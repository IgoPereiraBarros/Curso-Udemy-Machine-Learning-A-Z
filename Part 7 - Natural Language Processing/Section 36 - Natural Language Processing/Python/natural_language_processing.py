# -*- coding: utf-8 -*-

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
# quoting=3 -->  "String, String" -> "String", "string"
df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t', quoting=3)

# Cleaning the text
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(len(df)):
    reviews = re.sub('[^a-zA-Z]', ' ', df['Review'][i]).lower().split()
    reviews = [ps.stem(word) for word in reviews if word not in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    corpus.append(reviews)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

