#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[8]:


heart_data = pd.read_csv('/content/heart_disease_data.csv')


# In[9]:


heart_data.head()


# In[10]:


heart_data.tail()


# In[11]:


heart_data.shape


# In[12]:


heart_data.info()


# In[13]:


heart_data['target'].value_counts()


# In[14]:


X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']


# In[15]:


print(X)
print(Y)


# In[16]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[17]:


print(X.shape,X_train.shape,X_test.shape)


# In[18]:


model = LogisticRegression()


# In[19]:


model.fit(X_train,Y_train)


# In[20]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[21]:


print('training data accuracy :  ',training_data_accuracy)


# In[22]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[23]:


print('test data accuracy :  ',test_data_accuracy)


# In[24]:


input_data = (57,1,0,140,192,0,1,148,0,0.4,1,0,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The person does not have a heart disease')
else:
  print('The person has a heart disease')


# In[38]:


import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load some sample data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Save the model to a file
filename = 'my_classifier.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[39]:


filename = 'Heart_disease_model.sav'


# In[40]:


pickle.dump(classifier, open(filename, 'wb'))


# In[41]:


loaded_model = pickle.load(open('Heart_disease_model.sav','rb'))


# In[45]:


input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
scaler = StandardScaler()
scaler.fit(input_data_reshaped)
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:

import joblib
joblib.dump(model, 'model.pkl')



