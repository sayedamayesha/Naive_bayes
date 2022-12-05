#!/usr/bin/env python
# coding: utf-8

# Data attributes:
# 
# 1. Admit: 0-not admitted, 1- admitted
# 2. gre: gre score
# 3. Gpa: Gpa score
# 4. Rank: School rank
# 
# You need to create a machine learning model if a student is admitted or not given attributes. 
# 
# 1. Run logistic model and naive bayes. Compare the two results. Which one is better?
# 2. Create density plot of gre and gpa against admit.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns


# In[2]:


data= pd.read_csv(r'C:\CSV_files\Education.csv')


# In[3]:


data.head()


# In[4]:


data.isna().any()


# In[5]:


x= data.iloc[:,[1,2]].values


# In[6]:


y= data.iloc[:,3].values


# In[7]:


print(x)


# In[8]:


print(y)


# In[9]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)


# In[10]:


print(x_train)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


st_x= StandardScaler()


# In[13]:


x_train= st_x.fit_transform(x_train)


# In[14]:


x_test= st_x.transform(x_test)


# In[15]:


print(x_train)
print(x_test)


# In[16]:


classifier= LogisticRegression(random_state=0)


# In[17]:


classifier.fit(x_train, y_train)


# In[18]:


y_pred= classifier.predict(x_test)


# In[19]:


print(y_pred)


# In[20]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, x_test, y_test)


# In[24]:


data['admit'].value_counts()


# In[23]:


sns.countplot(x='admit', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')


# In[25]:


nfadmit = len(data[data['admit']==0])
nfattribute= len(data[data['admit']==1])
pct_of_admit= nfadmit/(nfadmit+nfattribute)
print("percentage of admit is", pct_of_admit*100)
pct_of_att = nfattribute/(nfadmit+nfattribute)
print("percentage of attribute", pct_of_att*100)


# In[27]:


#cm= confusion_matrix()


# In[29]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)


# In[30]:


y_pred= gnb.predict(x_test)


# In[31]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[32]:


data.plot.kde()


# In[33]:


data["gre"].plot.kde()


# In[35]:


data[ "gpa"].plot.kde()


# In[38]:


data[["gre", "gpa"]].plot.kde()


# In[ ]:




