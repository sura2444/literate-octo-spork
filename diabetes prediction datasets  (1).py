#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


# In[45]:


# Load the dataset
df=pd.read_csv('Downloads/diabetes_prediction_dataset.csv')
df


# In[46]:


# Explore the dataset
df.info()


# In[47]:


df.describe()


# In[48]:


# Handling missing values and data cleaning if necessary.
df.isnull().sum()


# In[49]:


df.columns


# In[50]:


cat_df = df.select_dtypes(include='object')
num_df = df.select_dtypes(exclude='object')
print("Categorical Features: ", cat_df.columns.to_list())
print("Numerical Features: ", num_df.columns.to_list())


# In[55]:


# Initialize LabelEncoders
gender_encoder = LabelEncoder()
smoking_history_encoder = LabelEncoder()

# Fit and transform the data
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['smoking_history'] = smoking_history_encoder.fit_transform(df['smoking_history'])

# Display the transformed dataset
print(df.head())


# In[56]:


#Visualizing distributions and relationships between features using plots.
df['diabetes'].value_counts()


# In[57]:


plt.figure(figsize=(10,8))
sns.heatmap(data=num_df.corr(),annot=True)
plt.show()


# In[58]:


# Let's define our X and y
# We will drop 'diabetes' from our X because it is our target variable
x= df.drop(['diabetes'],axis=1)
y= df['diabetes']
print(y)
print(x)


# In[66]:


#Splitting the dataset into training and testing sets.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)


# In[68]:


from sklearn.preprocessing import StandardScaler
# Initialize and fit the scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[70]:


from sklearn.linear_model import LogisticRegression
# Initialize the model with a higher max_iter
model = LogisticRegression(max_iter=1000)  
# Train the model
model.fit(x_train_scaled, y_train)


# In[73]:


from sklearn.metrics import classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(x_test_scaled)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[74]:


# Confusion Matrix
confusion_matrix(y_test, y_pred)


# In[75]:


import seaborn as sn
cm =confusion_matrix(y_test,y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
confusion_matrix(y_test, y_pred)


# In[ ]:




