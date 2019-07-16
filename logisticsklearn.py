#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[4]:


dfx=pd.read_csv('Logistic_X_Train.csv')
dfy=pd.read_csv('Logistic_Y_Train.csv')


# In[5]:


X=dfx.values
Y=dfy.values
print(X.shape)
print(Y.shape)


# In[6]:


Y=Y.reshape((-1,))


# In[9]:


logr=LogisticRegression(solver='lbfgs')


# In[10]:


logr.fit(X,Y)


# In[12]:


logr.predict([X[0]])


# In[13]:


Y[0]


# In[14]:


logr.score(X,Y)

logr.get_params(deep=True)
# In[20]:


logr.coef_


# In[21]:


logr.intercept_


# In[38]:


dXtest=pd.read_csv('Logistic_X_Test.csv')
Xtest=dXtest.values
n=Xtest.shape[0]
Xtest=np.array(Xtest)
print(Xtest.shape)


# In[40]:


Y=[]
import csv
with open('logisticpredicted.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    for i in range(n):
        y=logr.predict([Xtest[i]])
        writer.writerows([y])
        Y.append(y)
print(Y)
csvFile.close()


# In[52]:


coefficients=np.ndarray.tolist(logr.coef_)


# In[56]:


coefficients


# In[ ]:




