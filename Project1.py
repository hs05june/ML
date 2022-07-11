#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble  import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
housing = pd.read_csv('housing.csv',names=names)


# In[3]:


print(housing.head(5))


# In[4]:


housing['CHAS'].describe()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.hist(figsize=(30,30))


# In[7]:


train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(train_set)


# In[8]:


train_set_label = train_set['MEDV']
train_set = train_set.drop('MEDV',axis=1)


# In[9]:


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[10]:


housing_train = strat_train_set.drop('MEDV',axis=1)
housing_train_labels = strat_train_set['MEDV']
housing_test = strat_test_set.drop('MEDV',axis=1)
housing_test_labels = strat_test_set['MEDV']


# In[11]:


attributes = ['MEDV','RM','ZN','LSTAT','PTRATIO']
scatter_matrix(housing[attributes],figsize=(30,30))


# In[12]:


housing.plot(kind='scatter',y='MEDV',x='RM',alpha=0.8)
housing.plot(kind='scatter',y='MEDV',x='LSTAT',alpha=0.8)


# In[13]:


housing['TAXRM'] = housing['TAX']/housing['RM']
housing.plot(kind='scatter',x='TAXRM',y="MEDV",alpha=0.8)


# In[14]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])


# In[16]:


def result(scores):
    print("Scores :",scores)
    print("Mean :",scores.mean())


# In[17]:


my_pipeline.fit_transform(housing_train)
my_pipeline.fit_transform(housing_test)


# In[24]:


model = RandomForestRegressor()
# lr = DecisionTreeRegressor()
# lr.fit(housing_train,housing_train_labels)


# In[19]:


# pred_lr = lr.predict(housing_test)
# print(list(lr.predict(housing_test)))
# print(list(housing_test_labels))


# In[20]:


# variance = mean_squared_error(housing_test_labels,pred_lr)
# std = np.sqrt(variance)
# print(std)


# In[25]:


kfold = KFold(n_splits=10,random_state=42,shuffle=True)
score = cross_val_score(model,housing_train,housing_train_labels,cv=kfold,scoring='neg_mean_squared_error')
scores = np.sqrt(-score)
result(scores)


# In[ ]:




