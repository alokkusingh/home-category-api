#!/usr/bin/env python
# coding: utf-8

# In[54]:


import joblib


# In[55]:


model = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-mdl.joblib");
count_vect = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-vct.joblib");

model.predict(count_vect.transform(["cook"]).toarray())


# In[ ]:




