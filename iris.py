#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
Image(url='https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png', width=500)


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xg
from sklearn.model_selection import train_test_split


# In[3]:


#load and make the copy of Iris dataset to keep track of changes.
df = pd.read_csv('Iris1.csv') #read comma seperated values
df_copy=df.copy() #copy dataset


# In[4]:


#fetch first five rows from dataset
df_copy


# In[8]:


#Remove unnecessary feat from dataset Id
df_copy.drop(columns=['Id'],axis=0,inplace=True)


# In[9]:


#Check datatypes of each feat
df_copy.dtypes


# In[10]:


#check number of records and feilds present in dataset
df_copy.shape
print('Rows ---->',df.shape[0])
print('Columns ---->',df.shape[1])


# In[11]:


#see the descriptive statistics
df_copy.describe()


# In[12]:


#check the space complexicity taken by data
df_copy.size


# In[13]:


#checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed
df_copy.info()


# In[14]:


df_copy.columns = ['sl','sw','pl','pw','species']
df_split_iris=df_copy.species.str.split('-',n=-1,expand=True) #Remove prefix 'Iris-' from species col
df_split_iris.drop(columns=0,axis=1,inplace=True)#Drop 'Iris-' col
df_split_iris


# In[15]:


df3_full=df_copy.join(df_split_iris)
df3_full


# In[16]:


df3_full.rename({1:'species1'},axis=1,inplace=True) #Rename column
df3_full


# In[17]:


df3_full.drop(columns='species',axis=1,inplace=True) #Drop excessive column


# In[18]:


#final dataframe
df3_full


# In[20]:


from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
le.fit_transform(df3_full['species1'])
df3_full['species1']=le.fit_transform(df3_full['species1'])
df3_full


# In[21]:


x = df3_full.iloc[:,:-1]
x


# In[23]:


y = df3_full.iloc[:,-1]
y


# In[24]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=20)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
logi = LogisticRegression()
logi.fit(xtrain,ytrain)
logi_prediction = logi.predict(xtest)
logi_prediction


# In[28]:


print(logi.score(xtrain,ytrain)*100)
print(logi.score(xtest,ytest)*100)


# In[30]:


from sklearn.model_selection import GridSearchCV
para = {'penalty':['l1','l2','elasticnet'],
        'C':[1,2,3,4,5,6,10,20,30,40,50,1.5,2.3,1.6,1.9],
        'max_iter':[100,200,300,50,70,60,50]
        }


# In[31]:


classifier_logistic = GridSearchCV(logi,param_grid = para,scoring='accuracy',cv=5)


# In[34]:


classifier_logistic.fit(xtrain,ytrain)


# In[35]:


classifier_logistic.best_estimator_


# In[36]:


classifier_logistic.best_params_


# In[37]:


classifier_logistic.best_score_


# In[38]:


prediction = classifier_logistic.predict(xtest)
prediction


# In[39]:


from sklearn.metrics import accuracy_score,classification_report
grid_logi_accuracy_score1 = accuracy_score(ytest,prediction)
grid_logi_accuracy_score1=(np.round(grid_logi_accuracy_score1*100))
grid_logi_accuracy_score1


# In[40]:


confusion_matrix(ytest,prediction)


# In[41]:


class_pre_rec = classification_report(ytest,prediction)
print(class_pre_rec)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier(criterion='gini',
    splitter='best', 
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1,
    random_state=1,
    max_leaf_nodes=2,
    class_weight='balanced',
    ccp_alpha=0.01,)
tree_classifier.fit(xtrain,ytrain)


# In[47]:


tree_classifier.score(xtrain,ytrain)


# In[48]:


tree_classifier.score(xtest,ytest)


# In[50]:


tree_classifier.predict(xtest)


# In[51]:


tree_pred=tree_classifier.predict(xtest)


# In[52]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(ytest,tree_pred)


# In[53]:


print(classification_report(ytest,tree_pred))


# In[56]:


param_dict = {"criterion":['gini','entropy'],"max_depth":[1,2,3,4,5,6,7,None]}


# In[57]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(tree_classifier,param_grid=param_dict,n_jobs=-1)
grid


# In[58]:


grid.fit(xtrain,ytrain)


# In[59]:


grid.best_params_


# In[ ]:




