import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
import re

loaded_model = pickle.load(open("newsgrp_classifier_model.pkl", 'rb'))

features = []
infile = open('selected_features.txt', 'r')
for line in infile:
    features.append(line.strip())
infile.close()


def prediction(data):
    data = [data]
    pred_dataset = np.zeros([len(data), len(features)], int)

    for i in range(len(data)):
        words = data[i].lower()
        word = re.split(r'\W+', words)
        # Iterating over each word
        for j in word:
            # adding frequency corresponding to that word only which is in answer1(feature list)
            if j in features:
                pred_dataset[i][features.index(j)] += 1

    pred = loaded_model.predict(pred_dataset)
    return pred[0]




st.title("News Classifier")
st.write("I can only categorize news as Atheism, Automobiles, Motorcycles, Computer Graphics, Cryptography,Electronics, For Sale, Medicine, Politics, Religion, Space, Sports")
news_text = st.text_area("Enter News Here", "Please enter the entire article here")
if st.button("Classify"):
    prediiction = prediction(news_text)
    st.success("News Categorized as:: {}".format(prediiction))

