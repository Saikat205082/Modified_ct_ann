#!/usr/bin/env python
# coding: utf-8

# In[9]:





# In[ ]:


##Data


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df=pd.read_csv(r"E:\final_project4th\sem4data.csv")
df
data=df.fillna("Alive")
indexAge = data[ (data['Reason'] == 'Dropout')| (data['Reason'] == 'Not able to communicate')].index
data.drop(indexAge , inplace=True)


# In[10]:


data=df.fillna("Alive")
indexAge = data[ (data['Reason'] == 'Dropout')| (data['Reason'] == 'Not able to communicate')].index
data.drop(indexAge , inplace=True)

data


# In[11]:


p1=data[["T0","HR0","RR0","BPS0","BPD0","SPO20","FLOW0","T1","HR1","RR1","BPS1","BPD1","SPO21","FLOW1","Reason1","SMOKING","GENDER","FEVER","CHILLS","RIGOR","COUGH","EXPECTORATION","CHEST PAIN","CHEST DISCOMFORT","ANOSMIA","DYSGUESIA","NASAL BLOCK","RUNNING NOSE",
         "DIARRHOEA","CONSTIPATION","MALAISE","WEAKNESS","BODY ACHE","SOB","DM","HTN","COPD","HYPOTHYROID","ASTHMA","CKD","CABG"]]
p1
data1=pd.get_dummies(p1,columns =["SMOKING","GENDER","FEVER","CHILLS","RIGOR","COUGH","EXPECTORATION","CHEST PAIN","CHEST DISCOMFORT","ANOSMIA","DYSGUESIA","NASAL BLOCK","RUNNING NOSE",
         "DIARRHOEA","CONSTIPATION","MALAISE","WEAKNESS","BODY ACHE","SOB","DM","HTN","COPD","HYPOTHYROID","ASTHMA","CKD","CABG"])
#pd.set_option('display.max_rows', None)
data1


# In[ ]:





# In[12]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
X= data1.drop('Reason1', axis=1)
Y= data1['Reason1']
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split( 
          X, Y, test_size = 0.3, random_state =100)
dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(X_train,y_train)


# In[14]:


ypred=dt_classifier.predict(X_test)
print(ypred)
accuracy = metrics.accuracy_score(y_test, ypred)
print("Accuracy:", accuracy)


# In[15]:


from sklearn.metrics import precision_score, recall_score, f1_score



# Calculate precision, recall, and F1-score
precision = precision_score(y_test, ypred)
recall = recall_score(y_test, ypred)
f1 = f1_score(y_test, ypred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[16]:


feature_importances = dt_classifier.feature_importances_
feature_importances


# In[17]:


fi=pd.DataFrame({'Feature_name':X_train.columns,'importence':feature_importances})
fi=fi.sort_values("importence",ascending=False)
fi


# In[18]:


r=(dt_classifier.predict(X[["T1","HR1","RR1","BPS1","BPD1","SPO21","FLOW1","T0","HR0","RR0","BPS0","BPD0","SPO20","FLOW0","SMOKING_Current","SMOKING_Non Smoker","SMOKING_Past","GENDER_F","GENDER_M","FEVER_False","FEVER_True","CHILLS_False","CHILLS_True","RIGOR_False","RIGOR_True","COUGH_False","COUGH_True","EXPECTORATION_False","EXPECTORATION_True","CHEST PAIN_False","CHEST PAIN_True","CHEST DISCOMFORT_False",
                            "CHEST DISCOMFORT_True","ANOSMIA_False","ANOSMIA_True","DYSGUESIA_False","DYSGUESIA_True","NASAL BLOCK_False","NASAL BLOCK_True","RUNNING NOSE_False","RUNNING NOSE_True",
                            "DIARRHOEA_False","DIARRHOEA_True","CONSTIPATION_False","CONSTIPATION_True","MALAISE_False","MALAISE_True","WEAKNESS_False","WEAKNESS_True","BODY ACHE_False","BODY ACHE_True","SOB_False","SOB_True","DM_False","DM_True","HTN_False",
                            "HTN_True","COPD_False","COPD_True","HYPOTHYROID_False","HYPOTHYROID_True","ASTHMA_False","ASTHMA_True","CKD_False","CKD_True","CABG_False","CABG_True"]]))
r


# In[ ]:


# ANN data


# In[19]:


new=X[["SPO20","BPS1","HR1","GENDER_M","BPD1","RR1","FLOW1","SPO21","RUNNING NOSE_True","COUGH_False","FLOW0"]]
new


# In[27]:


new=pd.DataFrame(new)
new
Y=pd.DataFrame(Y)
new["Reason1"]=Y
new["CT_predicted"]=r
new
get_ipython().system('pip3 install keras')
get_ipython().system('pip3 install ann_visualizer')
get_ipython().system('pip install graphviz')
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
X1= new.drop('Reason1', axis=1)
Y1=new['Reason1']
pd.set_option('display.max_rows', None)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state =120)
print(X1_train.shape, X1_test.shape)
print(y1_train.shape, y1_test.shape)
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X1_train, y1_train, batch_size = 4, epochs = 10)
y_pred_prob = ann.predict(X1_test)
y_pred = (y_pred_prob > 0.5)
print("Accuracy Score", accuracy_score(y1_test, y_pred))
conf_matrix = confusion_matrix(y_true=y1_test, y_pred=y_pred)
conf_matrix
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
y_pred


# In[28]:


precision = precision_score(y1_test, y_pred)
recall = recall_score(y1_test, y_pred)
f1 = f1_score(y1_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:





# In[ ]:




