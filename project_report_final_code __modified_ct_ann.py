#!/usr/bin/env python
# coding: utf-8

# In[45]:


##data 


# In[1]:


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
data=df.fillna("Alive")
indexAge = data[ (data['Reason'] == 'Dropout')| (data['Reason'] == 'Not able to communicate')].index
data.drop(indexAge , inplace=True)
p1=data[["T0","HR0","RR0","BPS0","BPD0","SPO20","FLOW0","T1","HR1","RR1","BPS1","BPD1","SPO21","FLOW1","Reason1","SMOKING","GENDER","FEVER","CHILLS","RIGOR","COUGH","EXPECTORATION","CHEST PAIN","CHEST DISCOMFORT","ANOSMIA","DYSGUESIA","NASAL BLOCK","RUNNING NOSE",
         "DIARRHOEA","CONSTIPATION","MALAISE","WEAKNESS","BODY ACHE","SOB","DM","HTN","COPD","HYPOTHYROID","ASTHMA","CKD","CABG"]]
p1
data1=pd.get_dummies(p1,columns =["SMOKING","GENDER","FEVER","CHILLS","RIGOR","COUGH","EXPECTORATION","CHEST PAIN","CHEST DISCOMFORT","ANOSMIA","DYSGUESIA","NASAL BLOCK","RUNNING NOSE",
         "DIARRHOEA","CONSTIPATION","MALAISE","WEAKNESS","BODY ACHE","SOB","DM","HTN","COPD","HYPOTHYROID","ASTHMA","CKD","CABG"])
#pd.set_option('display.max_rows', None)
data1
X= data1.drop('Reason1', axis=1)
y= data1['Reason1']
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data


# In[ ]:


## modified ct model code


# In[20]:


import numpy as np
from itertools import combinations
from scipy.stats import spearmanr, pointbiserialr, kendalltau
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]

        if depth == self.max_depth or all(samples == 0 for samples in num_samples_per_class):
            return TreeNode(value=np.argmax(num_samples_per_class))

        best_gini = float('inf')
        best_feature_index = None
        best_threshold = np.inf  # Initialize to infinity

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = ~left_indices

        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)

    def _gini_impurity(self, y_left, y_right):
        num_left = len(y_left)
        num_right = len(y_right)
        total_samples = num_left + num_right

        gini_left = 1.0 - sum((np.sum(y_left == c) / num_left) ** 2 for c in range(self.num_classes))
        gini_right = 1.0 - sum((np.sum(y_right == c) / num_right) ** 2 for c in range(self.num_classes))

        gini = (num_left / total_samples) * gini_left + (num_right / total_samples) * gini_right
        return gini

    def remove_correlated_features(self, X, y, feature_names, parent_feature_index):
        remaining_feature_indices = list(range(len(feature_names)))
        remaining_feature_names = list(feature_names)

        # Calculate correlations with the parent feature
        parent_feature = X[:, parent_feature_index]
        correlations = []
        for i in remaining_feature_indices:
            correlation = self.compute_correlation(parent_feature, X[:, i])
            correlations.append((i, correlation))

        # Filter out highly correlated features
        filtered_indices = [i for i, corr in correlations if abs(corr) >0.6]  # Adjust the correlation threshold as needed

        # Create new dataset with filtered features
        X_filtered = X[:, filtered_indices]
        new_feature_names = [feature_names[i] for i in filtered_indices]

        # Fit a new decision tree on the filtered dataset
        self.fit(X_filtered, y)

        # Return the filtered dataset and new feature names
        return X_filtered, new_feature_names

    def compute_correlation(self, feature1, feature2):
        # Spearman correlation for numeric-numeric
        if all(isinstance(val, (int, float)) for val in feature1) and all(isinstance(val, (int, float)) for val in feature2):
            return spearmanr(feature1, feature2)[0]

        # Point-biserial correlation for dummy-numeric
        elif set(feature1) == {0, 1} and all(isinstance(val, (int, float)) for val in feature2):
            return pointbiserialr(feature1, feature2)[0]

        # Point-Kendal tau correlation for numeric-dummy
        elif set(feature2) == {0, 1} and all(isinstance(val, (int, float)) for val in feature1):
            return kendalltau(feature2, feature1)[0]

        # No correlation for other cases
        else:
            return 0

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while node.left:
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions

    def print_important_features(self, feature_names):
        # Print important features after fitting the decision tree
        # In this simple implementation, we print the feature names of all nodes in the decision tree
        self._print_tree_features(self.tree, feature_names)

    def _print_tree_features(self, node, feature_names):
        if node is not None:
            if node.feature_index is not None:
                print("Feature:", feature_names[node.feature_index])
            self._print_tree_features(node.left, feature_names)
            self._print_tree_features(node.right, feature_names)

# Example X_train with mixed data types


# In[29]:


# Load the simulated dataset
#simulated_data = pd.read_csv('simulated_data.csv')

# Separate features (X) and target variable (y)
#X = simulated_data.drop('response', axis=1)
#y = simulated_data['response']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=49)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)

# Fit the classifier
clf.fit(X_train.values, y_train.values)


# In[30]:



predictions = clf.predict(X.values)
predictions1 = clf.predict(X_test.values)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, predictions1)
print("Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, predictions1)
print("Confusion Matrix:")
print(conf_matrix)


# In[31]:


from sklearn.metrics import precision_score, recall_score, f1_score



# Calculate precision, recall, and F1-score
precision = precision_score(y_test, predictions1)
recall = recall_score(y_test, predictions1)
f1 = f1_score(y_test, predictions1)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[32]:


clf.print_important_features(X.columns)


# In[ ]:


## ANN data preperartion


# In[33]:


new=X[["FLOW0","BPS0","T1","BPD0","HR1","T0"]]
 


# In[67]:


new=pd.DataFrame(new)
new
y=pd.DataFrame(y)
new["Reason1"]=y
new["CT_predicted"]=predictions 
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


# In[14]:


new


# In[ ]:


##ANN code


# In[15]:


new=pd.DataFrame(new)

Y=pd.DataFrame(y)
new["Reason1"]=Y
new["CT_predicted"]=predictions
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


# In[16]:


X1.shape


# In[21]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming X1 and Y1 are your features and labels respectively

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y1, test_size=0.2, random_state=190)

# Build the ANN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(640, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3, verbose=2)

# Predict on the test data
y_pred = model.predict(X_test)

# Convert probabilities to classes
y_pred_classes = (y_pred > 0.5).astype("int32")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)


# In[23]:


from sklearn.metrics import precision_score, recall_score, f1_score



# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




