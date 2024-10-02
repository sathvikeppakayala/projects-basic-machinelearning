#!/usr/bin/env python
# coding: utf-8

# # Email Spam Filtering using Naive Bayes in machine learning
# ## we have taken a sample dataset available on internet which has spam and ham mails available at my gihub repository - Machine Learning @sathvikeppakayala
# ## the columns Available in dataset are
# ## my github - sathvikeppakayala
# - Messsage ID
# - message
# - spam/ham

# In[50]:


# importing all the required Libraires
import pandas as pd # aliasing pandas with pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[25]:


# reading the dataset - name of the dataset is spam.csv
data = pd.read_csv('spam.csv')
data = data.iloc[:, :-3]
print(data)


# In[26]:


# now we will categorize the data based on spam/ ham
data.groupby('v1').describe()


# In[28]:


# we simple make spam return 1 and ham return 0 using apply method
# we create a spam row 
data['spam'] = data['v1'].apply(lambda x: 1 if x == "spam" else 0)
data


# In[44]:


# now from sklearn we import model_selection for split the issues
x_train, x_test, y_train, y_test = train_test_split(data.v2, data.spam, test_size = 0.30 )


# In[48]:


# now we use CountVectorizer
cv = CountVectorizer()
x_train_cnt = cv.fit_transform(x_train.values)
x_train_cnt.toarray()[:4] #we transfrom to array dimension of 4


# # we use Naive Bayes 
# ## Naive Bayes are three types
# - Bernoulli Naive Bayes - used for only Binary 0, 1
# - Multinomial Naive Bayes - used for discrete data
# -  Guassian Naive Bayes - used for Nomal Distribution
# # We use Multinomial Naive Bayes as we Imported earlier

# In[52]:


model = MultinomialNB()
model.fit(x_train_cnt, y_train)


# # Now Let's test with a sample emails

# In[57]:


emails = [
    "Hi Eppakayala, Your futures order was completed!", 
    "A Samsung account login may be required to access certain AI features. Samsung promotes responsible use of AI features. S Pen is available with select models only.*Pre-reserve period starts from 17th Sept'24 to 25 Sept'24. #E-store voucher benefits worth ₹ 3499 redeemable only on 1 unit of 45W Travel Adapter on Samsung Shop (i.e. Samsung.com or Samsung Shop app). For detailed terms and conditions for Pre-reserve made on Samsung Shop", 
    "Dear Intern, Congratulations🎉. You are Selected for TechnoHacks EduTech Internship Program 2024, Batch 57"
]
email_cnt = cv.transform(emails)
model.predict(email_cnt)
# here 0 is not a spam and 1 is a spam email and the output is same as my email spam folder spam and ham


# # Now we test the accuracy

# In[58]:


x_test_cnt = cv.transform(x_test)
model.score(x_test_cnt, y_test)


# # Now we perform the same using sklearn.pipeline

# In[63]:


from sklearn.pipeline import Pipeline
cnt = Pipeline([("vectorizer", CountVectorizer()),
               ("naivebayes", MultinomialNB())])


# In[64]:


cnt.fit(x_train, y_train)


# In[65]:


cnt.score(x_test, y_test)


# In[67]:


cnt.predict(emails)

