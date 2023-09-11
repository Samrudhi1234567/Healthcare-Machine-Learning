#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install seaborn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#!pip install tensorflow
import tensorflow as tf


# # READING THE DATASET

# In[2]:


dataset = pd.read_csv(r'C:\Users\SHREE\Downloads\Heart Failure Dataset.csv')
dataset.head()


# In[3]:


dataset.shape


# In[4]:


dataset.info()


# # CHECKING FOR NULL VALUES

# In[5]:


dataset.isnull().sum()


# # CORRELATION GRAPH

# In[6]:


plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Graph');


# In[7]:


dataset.hist()


# # SCALING THE COLUMS

# In[8]:


columns_to_scale = ['age', 'education', 'currentSmoker', 'BPMeds', 'sysBP','diaBP','diabetes','totChol','BMI']
Scaler = StandardScaler()
dataset[columns_to_scale] = Scaler.fit_transform(dataset[columns_to_scale])


# # DEFINING THE DATASET FOR TRAINING

# In[9]:


X = dataset.drop('TenYearCHD', axis=1)
y = dataset['TenYearCHD']
X = dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
print(X.shape)
print(y.shape)


# # IMPUTING TO DEAL WITH NULL VALUES

# In[10]:


from sklearn.impute import SimpleImputer
_ = SimpleImputer(missing_values = np.nan, strategy = 'mean')
_.fit(X)
X = _.transform(X)


# # DEFINING AND SPLITTING THE TRAINING AND TESTING DATA

# In[11]:


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=0)


# # THE SHAPE OF THE TRAINING AND TESTING DATA

# In[12]:


print("shape x_train:", Xtr.shape)
print("shape y_train:", ytr.shape)
print("shape x_test:", Xte.shape)
print("shape y_test:", yte.shape)


# # SCALER FUNCTION

# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.fit_transform(Xte)


# # TRAINING THE MODEL AND GETTING THE REPORTS FOR IT
# This is the custom function that first trains the model and then gets report for it

# In[14]:


def report(clf):
    clf.fit(Xtr,ytr)
    y_pred = clf.predict(Xte)
    score = cross_val_score(clf,Xtr,ytr,cv=5).mean()
    print("Accuracy of the model",clf,"=",accuracy_score(yte,y_pred))
    print("Cross val score:",score)
    print("Confusion Matrix:")
    print(confusion_matrix(yte,y_pred))
    print("Classification report: \n",classification_report(yte, y_pred))
    matrix = plot_confusion_matrix(clf, Xte, yte)
    matrix.ax_.set_title('Confusion Matrix')
    plt.show()


# # LOGISTIC REGRESSION

# In[15]:


clf = LogisticRegression(solver='saga',max_iter=100,tol=0.0001,C=1)
report(clf)


# # DECISION TREE CLASSIFIER

# In[16]:


clf = DecisionTreeClassifier(max_features='auto',min_samples_leaf=2)
report(clf)


# # RANDOM FOREST CLASSIFIER 

# In[17]:


clf = RandomForestClassifier(criterion='gini',class_weight='balanced',max_features='log2')
report(clf)


# # KNEIGHBORS CLASSIFIER 

# In[18]:


clf = KNeighborsClassifier(algorithm = 'auto',n_neighbors=6)
report(clf)


# # GAUSSIAN NAIVE BAYES

# In[19]:


clf = GaussianNB()
report(clf)


# # SUPPORT VECTOR MACHINE

# In[20]:


clf = SVC(gamma='auto',kernel='poly')
report(clf)


# # Artificial Neural Networks(ANN) using tensorflow

# In[30]:


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 4, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 4, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 1,activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model = ann.fit(Xtr,ytr,validation_data=(Xte,yte), batch_size = 16,epochs=100)


# In[31]:


y_pred = ann.predict(Xte)
y_pred = (y_pred > 0.5)
c_m = confusion_matrix(yte, y_pred)
print(c_m)
print(accuracy_score(yte, y_pred))
print(classification_report(yte, y_pred))


# In[32]:


plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


# In[ ]:




