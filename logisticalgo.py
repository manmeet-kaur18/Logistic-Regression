#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dfx=pd.read_csv('Logistic_X_Train.csv')
dfy=pd.read_csv('Logistic_Y_Train.csv')


# In[3]:


X=dfx.values


# In[4]:


Y=dfy.values


# In[5]:


print(X.shape)


# In[6]:


print(Y.shape)


# In[7]:


y_train=Y.reshape((-1,))


# In[8]:


print(y_train.shape,Y.shape)


# In[14]:


plt.scatter(X[:,0],X[:,1])


# In[15]:


plt.scatter(X[:,0],X[:,2])


# In[ ]:


def hypothesis(x,w,b):
    h=np.dot(x,w)+b
    print(h)
    return sigmoid(h)

def sigmoid(z):
    z1=1.0/(1.0 + np.exp(-z))
#     print("sigmoid")
#     print(z1)
    return z1

def error(y_true,x,w,b):
    m=x.shape[0]
    err=0.0
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        err += y_true[i]*np.log(hx) - (1-y_true[i])*np.log(1-hx)
#         print(err)
    return -err/m

def get_grads(y_true,x,w,b):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    
    m=x.shape[0]
    for i in range(m):
        hx=hypothesis(x[i],w,b)
        
        grad_w += -1*(y_true[i]-hx)*x[i]
        grad_b = -1*(y_true[i]-hx)
        
    grad_w /= m
    grad_b /=m
    
    return [grad_w,grad_b]

def grad_descent(x,y_true,w,b,learning_rate=0.1):
    
    err=error(y_true,x,w,b)
    [grad_w,grad_b] = get_grads(y_true,x,w,b)
    
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    
    return err,w,b    


# In[ ]:


loss=[]
acc=[]

w = 2*np.random.random((X.shape[1],))
b = 5*np.random.random()
for i in range(10):
    l,w,b = grad_descent(X,y_train,w,b,learning_rate=0.01)
    loss.append(l)


# In[ ]:


print(X[5])
print(loss)
# plt.plot(loss)


# In[ ]:




