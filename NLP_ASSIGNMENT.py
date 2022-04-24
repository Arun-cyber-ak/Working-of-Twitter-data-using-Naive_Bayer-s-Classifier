#!/usr/bin/env python
# coding: utf-8

# NAME: ARUN KUMAR M
# 
# ROLLNO.:215229104

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


df.head(20)


# In[4]:


df1 = pd.read_csv("test.csv")


# In[5]:


df1


# ## Data Cleaning

# **Changing the text data into lowercase**

# In[6]:


df["text"] = df["text"].apply(lambda x: x.lower())
df1["text"] = df1["text"].apply(lambda x: x.lower())


# **Removing the special character in the text**

# In[7]:


df['text']= df['text'].str.replace('rt ',"").str.replace('@','').str.replace('#','').str.replace('[^\w\s]','').str.replace('[1-9]','')
df1['text']= df1['text'].str.replace('rt ',"").str.replace('@','').str.replace('#','').str.replace('[^\w\s]','').str.replace('[1-9]','')


# **Removing the URL link in the text**

# In[8]:


import re


# In[9]:


def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

df["text"] = df["text"].apply(lambda x: remove_URL(x))
df1["text"] = df1["text"].apply(lambda x: remove_URL(x))


# **Removing HTML tags in the text**

# In[10]:


def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

df["text"] = df["text"].apply(lambda x: remove_html(x))
df1["text"] = df1["text"].apply(lambda x: remove_html(x))


# **Removing the punctuation in the text**

# In[11]:


import string


# In[12]:


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))


df["text"] = df["text"].apply(lambda x: remove_punct(x))
df1["text"] = df1["text"].apply(lambda x: remove_punct(x))


# #### After Cleaning the data 

# In[13]:


df[['text']].head(20)


# ## Data Pre-Processing

# **Dropping of unnecessary columns**

# In[14]:


df = df.drop('keyword', axis=1)
df1 = df1.drop('keyword', axis=1)


# In[15]:


df = df.drop('location', axis=1)
df1 = df1.drop('location', axis=1)


# **Tokenization**

# In[16]:


import nltk
from nltk import TweetTokenizer

tokenizer = TweetTokenizer()

df['tokens'] = [tokenizer.tokenize(item) for item in df.text]
df1['tokens'] = [tokenizer.tokenize(item) for item in df1.text]


# **Lemmatization**

# In[17]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_item(item):
    new_item = []
    for x in item:
        x = lemmatizer.lemmatize(x)
        new_item.append(x)
    return " ".join(new_item)

df['tokens'] = [lemmatize_item(item) for item in df.tokens]
df1['tokens'] = [lemmatize_item(item) for item in df1.tokens]


# **Vectorization**

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


X = vectorizer.fit_transform(df.text).toarray()
y = df['target']


# In[19]:


df[['text']]


# #### Spliting  the data for training a model

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# **Creating a Model**

# In[21]:


from sklearn.naive_bayes import MultinomialNB


# In[22]:


mb = MultinomialNB()
mb.fit(X_train,y_train)


# In[23]:


y_pred = mb.predict(X_test)


# **Printing the accuracy,confusion matrix and classification**

# ### Accuracy

# In[24]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[25]:


acc = accuracy_score(y_test,y_pred)
print("The accuracy of the model is :",acc)


# ### Confusion matrix

# In[26]:


cf = confusion_matrix(y_test,y_pred)


# In[27]:


cf


# ### Classification Report

# In[28]:


print(classification_report(y_test,y_pred))


# In[ ]:




