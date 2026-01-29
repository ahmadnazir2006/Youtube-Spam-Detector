import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import emoji as emo
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

print("Hello from app.py")
def clean_data(comment): #this function is for cleaning the data by removing the url, emojis, and special characters
    text=str(comment).lower()
    url_pattern = r'(https?://\S+|www\.\S+|watch\?v=\S+)'
    text = re.sub(url_pattern, 'link_token', text)
    emo.demojize(text)
    text=text.replace('@', 'at_token')
    text=text.replace('#', 'hash_token')
    text=text.replace('&', 'and_token')
    text=text.replace('*', 'star_token')
    text=text.replace(':',' ')
    text = " ".join(text.split())
    return text
def preprocess_data(df):#this function is for dropping the unnecessary columns and applying the clean_data function to the content column
    df=df[['CONTENT','CLASS']].copy()
    df['CONTENT']=df['CONTENT'].apply(clean_data)
    return df
df=pd.read_csv('Youtube01-Psy.csv')
df=preprocess_data(df)
print(df.head())


train_X,temp_X,train_y,temp_y=train_test_split(
    df['CONTENT'],
    df['CLASS'],
    test_size=0.4, #here we are spliiting in two steps first by 60% and then splitting the remaining 40% into valid and test by 50% each
    random_state=42,
    stratify=df['CLASS'] #this ensures that both spam and ham cases for training are equally divided in the process of splitting
)

valid_X,test_X,valid_y,test_y=train_test_split(
    temp_X,
    temp_y,
    test_size=0.5,
    random_state=42,
    stratify=temp_y)
print(f"Train: {len(train_X)}, Valid: {len(valid_X)}, Test: {len(test_X)}")

vectorizer=TfidfVectorizer(stop_words='english',max_features=2000)#this is for converting text to numbers for model by dedicating them a specific weightage
train_X_vec=vectorizer.fit_transform(train_X)#used fit only here :fit allows model to learn/read from data
valid_X_vec=vectorizer.transform(valid_X)    #used transform here only so model learns from same data
test_X_vec=vectorizer.transform(test_X)
print(f"Train: {train_X_vec.shape}, Valid: {valid_X_vec.shape}, Test: {test_X_vec.shape}")
model=MultinomialNB()
model.fit(train_X_vec,train_y)
valid_pred=model.predict(valid_X_vec)
valid_acc=accuracy_score(valid_y,valid_pred)
print(f"--- AI Training Result ---")
print(f"Validation Accuracy: {valid_acc * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(valid_y,valid_pred))


import joblib

# This saves trained model and your vectorizer dictionary
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Files saved successfully!")