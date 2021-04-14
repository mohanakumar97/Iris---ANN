#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Necessary Libraries :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp


# In[2]:


# Import the Data set :

iris = pd.read_csv('iris.csv')
iris


# In[3]:


iris.describe()


# In[4]:


iris.info()


# In[5]:


iris.shape


# In[6]:


# Null values in the Dataset :

iris.isnull().sum()


# In[7]:


iris = iris.drop('Id',axis=1)


# In[8]:


# Label Encodeing :

from sklearn.preprocessing import LabelEncoder
LB = LabelEncoder()


# In[9]:


iris['Species']= LB.fit_transform(iris['Species'])
iris.head()


# In[30]:


pp.ProfileReport(iris)


# In[10]:


# Independent Variables :

ind_x = iris.drop('Species',axis=1).values
ind_x


# In[11]:


# Dependent Variable :
dep_y = pd.get_dummies(iris['Species']).values
dep_y


# In[12]:


# Train test split model :

from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(ind_x,dep_y, test_size = 0.2, random_state = 1)


# In[14]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[15]:


# Normalizing the train and test :

from sklearn.preprocessing import StandardScaler
normalize = StandardScaler()

x_train = normalize.fit_transform(x_train)
x_test = normalize.fit_transform(x_test)


# In[16]:


# Layers :

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten

model = Sequential()
model.add(Dense(4,input_dim=4,activation = 'relu'))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(3,activation = 'softmax'))


# In[17]:


# Compile :

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[18]:


model.fit(x_train, y_train, batch_size = 6, epochs = 50)


# In[25]:


# Metrics :

from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


y_pred = model.predict_classes(x_test)
y_test_classes = np.argmax(y_test,axis=1)
cm = print(confusion_matrix(y_test_classes,y_pred))


# In[21]:


print(classification_report(y_test_classes,y_pred))


# In[26]:


# Hence the Model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




