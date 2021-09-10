#!/usr/bin/env python
# coding: utf-8

# ### Flood Control System
# - Predicts the risk of floods based on monthly average rainfall.
# - Dataset used is Kerala Dataset from Kaggle.

# In[1]:


# Installing Libraries: Importing modules from packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings


# In[2]:


# Data Gathering: Importing the dataset 

data = pd.read_csv(r'C:\path\Dataset.csv')


# In[3]:


# Data Exploration: Drawing insights from data

data.head(10)


# In[4]:


# Data Exploration: Determining the shape of dataset

data.shape


# In[5]:


# Data Exploration: Summarizing the data 

data.describe()


# In[6]:


# Data Cleaning: Checking for missing values

data.info() 


# In[7]:


# Data Pre-Processing: 
#  FLOODS  YES -> 1
#  FLOODS  NO -> 0

data['FLOODS'].replace(['YES','NO'],[1,0],inplace=True)
data.head(10)


# In[8]:


# Data Exploration: Correlation between Monthly-Rainfall and Flood

corel = data.iloc[:,[*range(2,14),15]].corr()
corel["FLOODS"]


# In[9]:


# Data Exploration: Visualizing Heatmap

plt.figure(figsize=(10,10))
dataplot = sns.heatmap(corel, annot = True, fmt='.2g', mask = np.triu(corel),cmap= 'coolwarm',cbar_kws= {'orientation': 'horizontal'})
plt.show()


# In[10]:


# Data Visualization: Monthly average rainfall

x1 = data.iloc[:,2:14]
y1=x1.mean()

ax = y1.plot.barh(figsize=(15,8),colormap='Set2')
plt.xlabel('Rainfall (in mm)',fontsize=20)
plt.ylabel('Month',fontsize=20)
 
# Adding x, y gridlines
ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5, alpha = 0.2)

# Displaying top values
ax.invert_yaxis()

# Adding annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2,i.get_y()+0.3,str(round((i.get_width()), 2)),fontsize = 10,fontweight ='bold',color ='grey')


# In[11]:


# Dividing the dataset into dependent and independent variables

x = data.iloc[:,2:14]
y = data.iloc[:, -1]


# In[12]:


x.head()


# In[13]:


y.head()


# In[14]:


# Splitting the data into Training and Testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

# Determining the shapes of training and testing sets

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[15]:


# Standard Scaling: Scaling the data for optimised predictions 

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# ***

# ## Train the Model

# ### Random Forest 

# In[16]:


# Creating the model
rfc = RandomForestClassifier(n_estimators = 250,oob_score = True)

# Feeding the training set into the model
rfc.fit(x_train, y_train)

# Predicting the results for the test set
pred_rfc = rfc.predict(x_test)

print(classification_report(y_test, pred_rfc))


# In[17]:


x_train


# In[18]:


# Creating the model
rfc = RandomForestClassifier(n_estimators = 250,oob_score = True)

# Feeding the training set into the model
rfc.fit(x_train, y_train)

# Predicting the results for the test set
pred_rfc = rfc.predict(x_test)

print(classification_report(y_test, pred_rfc))


# ###  Decision Tree

# In[19]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
pred_dtc = dtc.predict(x_test)
print(classification_report(y_test, pred_dtc))


# ### Support Vector Machine

# In[20]:


svc = SVC()
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
print(classification_report(y_test, pred_svc))


# ### K-Nearest Neighbors 

# In[21]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
pred_knn=knn.predict(x_test)
print(classification_report(y_test, pred_knn))


# ### Logistic Regression 

# In[22]:


lor = LogisticRegression()
lor.fit(x_train, y_train)
pred_lor = lor.predict(x_test)
print(classification_report(y_test, pred_lor))


# In[23]:


# Conclusion: Comparing the results!

conclusion = pd.DataFrame({'Model': ["Random Forest","K-Nearest Neighbors","Logistic Regression","Decision Tree","Support Vector Machine"],
                           'Accuracy': [accuracy_score(y_test,pred_rfc),accuracy_score(y_test,pred_knn),
                                    accuracy_score(y_test,pred_lor),accuracy_score(y_test,pred_dtc),accuracy_score(y_test,pred_svc)]})
conclusion


# In[24]:


# Visualizing Results

plt.subplots(figsize=(10, 5))
axis = sns.barplot(x = 'Model', y = 'Accuracy', data =conclusion, palette="mako" )
axis.set(xlabel='Model', ylabel='Accuracy')

# Adding annotation to bars
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# ***

# ## Calculating  _ROC_Score_ for the top 3 models

# In[25]:


# Random Forest

# Calculating Training and Testing accuracies 
print("RF Training accuracy :", rfc.score(x_train, y_train))
print("RF Testing accuracy :", rfc.score(x_test, y_test))

# Calculating ROC Score
print("RF ROC score:%f"%(roc_auc_score(y_test,pred_rfc)*100))

# Confusion Matrix
print(confusion_matrix(y_test, pred_rfc))


# In[26]:


#Logistic Regression

# Calculating Training and Testing accuracies 
print("LR Training accuracy :", lor.score(x_train, y_train))
print("LR Testing accuracy :", lor.score(x_test, y_test))

# Calculating ROC Score
print("LR ROC score:%f"%(roc_auc_score(y_test,pred_lor)*100))

# Confusion Matrix
print(confusion_matrix(y_test, pred_lor))


# In[37]:


# SVM

# Calculating Training and Testing accuracies 
print("SVM Training accuracy :", svc.score(x_train, y_train))
print("SVM Testing accuracy :", svc.score(x_test, y_test))

# Calculating ROC Score
print("SVM ROC score:%f"%(roc_auc_score(y_test,pred_svc)*100))

# Confusion Matrix
print(confusion_matrix(y_test, pred_svc))


# In[28]:


# Model Evaluation: SVM model using Cross Validation

model_eval = cross_val_score(estimator = svc, X = x_train, y = y_train, cv = 10)
model_eval.mean()


# In[29]:


# Model Evaluation: RFC model using Cross Validation

model_eval = cross_val_score(estimator = rfc, X = x_train, y = y_train, cv = 10)
model_eval.mean()


# In[30]:


# Data Accuracy: Tabulating Actual vs Predicted values for RFC 

y_test = np.array(list(y_test))
y_pred = np.array(pred_rfc)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred_rfc.flatten()})
df[:20]


# In[31]:


# Data Accuracy Visualization: Constructing Barplot of the above response       

df1 = df.head(20)
df1.plot(kind='bar',figsize=(10,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ***

# ## Deep Learning 

# ### Multi Layer Perceptron 

# In[32]:


# Creating the model
model = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 150)

# Feeding the training data to the model
model.fit(x_train, y_train)

# Calculating the accuracies
print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))


# ### Artificial Neural Network 

# In[33]:


# Creating the model
model = Sequential()

# First hidden layer
model.add(Dense(8, activation = 'relu'))

# Second hidden layer
model.add(Dense(8, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Building the model:
history = model.fit(x_train, y_train, batch_size=25, epochs=100, verbose=2, validation_data=(x_test, y_test))


# In[34]:


# Visualizing Loss / Epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()


# In[35]:


# Visualizing Accuracy / Epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('Epoch ')
plt.show()


# In[ ]:




