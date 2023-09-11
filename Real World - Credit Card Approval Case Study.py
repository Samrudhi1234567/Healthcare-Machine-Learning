#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_excel(r"C:\Users\SHREE\Desktop\Samrudhi\sam\Credit approval classification case study.xlsx")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


data.describe(include = "all")


# In[10]:


data2 = data.fillna({"A1" : "b", "A2" : 32, "A4" : "u", "A5" : "g", "A6" : "c", "A7" : "v", "A14" : 184})


# In[11]:


data2.head()


# In[12]:


data2.describe(include = "all")


# In[13]:


data2.info()


# In[14]:


data2.plot (kind = "box")


# In[15]:


data2["A2"].plot (kind = "box")


# In[16]:


data2.describe()


# In[17]:


iqr_a2 = 37.70-22.67


# In[18]:


iqr_a2


# In[19]:


Uledge_a2 = 37.70+(1.5*15.03)


# In[20]:


Uledge_a2


# In[21]:


A2_dummy = []
for i in data2["A2"]:
    if i > 60.24:
        A2_dummy.append(60.24)
    else:
        A2_dummy.append(i)


# In[22]:


data2["A2_dummy"] = A2_dummy


# In[23]:


data2["A2_dummy"].plot (kind = "box")


# In[24]:


data2.columns


# In[25]:


iqr_a3 = 7.20-1


# In[26]:


uledge_a3 = 7.20+1.5*iqr_a3


# In[27]:


uledge_a3


# In[28]:


A3_dummy = []
for i in data["A3"]:
    if i > 16.5:
        A3_dummy.append(16.5)
    else:
        A3_dummy.append(i)


# In[29]:


data2["A3_dummy"] = A3_dummy


# In[30]:


data2["A3_dummy"].plot (kind = "box")


# In[31]:


iqr_a8 = 2.62-0.16


# In[32]:


uledge_a8 = 2.62+1.5*iqr_a8


# In[33]:


uledge_a8


# In[34]:


A8_dummy = []
for i in data2["A8"]:
    if i > 6.3:
        A8_dummy.append(6.3)
    else:
        A8_dummy.append(i)


# In[35]:


data2["A8_dummy"] = A8_dummy


# In[36]:


data2["A8_dummy"].plot (kind = "box")


# In[37]:


uledge_a11 = 3+1.5*3


# In[38]:


uledge_a11


# In[39]:


A11_dummy = []
for i in data2["A11"]:
    if i > 7.5:
        A11_dummy.append(7.5)
    else:
        A11_dummy.append(i)


# In[40]:


data2["A11_dummy"] = A11_dummy


# In[41]:


data2["A11_dummy"].plot (kind = "box")


# In[42]:


iqr_a14 = 272-80


# In[43]:


uledgea14 = 272+1.5*iqr_a14


# In[44]:


uledgea14


# In[45]:


A14_dummy = []
for i in data2["A14"]:
    if i > 560:
        A14_dummy.append(560)
    else:
        A14_dummy.append(i)


# In[46]:


data2["A14_dummy"] = A14_dummy


# In[47]:


data2["A14_dummy"].plot (kind = "box")


# In[48]:


uledge_a15 = 395+1.5*395


# In[49]:


uledge_a15


# In[50]:


A15_dummy = []
for i in data2["A15"]:
    if i > 987:
        A15_dummy.append(987)
    else:
        A15_dummy.append(i)


# In[51]:


data2["A15_dummy"] = A15_dummy


# In[52]:


data2["A15_dummy"].plot (kind = "box")


# In[53]:


data2.plot (kind = "box")


# In[54]:


from sklearn.preprocessing import LabelEncoder


# In[55]:


lb_make = LabelEncoder()


# In[56]:


data2["A1_N"] = lb_make.fit_transform(data2["A1"])
data2["A4_N"] = lb_make.fit_transform(data2["A4"])
data2["A5_N"] = lb_make.fit_transform(data2["A5"])
data2["A6_N"] = lb_make.fit_transform(data2["A6"])
data2["A7_N"] = lb_make.fit_transform(data2["A7"])
data2["A9_N"] = lb_make.fit_transform(data2["A9"])
data2["A10_N"] = lb_make.fit_transform(data2["A10"])
data2["A12_N"] = lb_make.fit_transform(data2["A12"])
data2["A13_N"] = lb_make.fit_transform(data2["A13"])
data2["A16_N"] = lb_make.fit_transform(data2["A16"])


# In[57]:


data2.head()


# In[58]:


data2.info()


# In[59]:


data2.dtypes


# In[60]:


data2_model = data2[["A1_N", "A2_dummy", "A3_dummy", "A4_N", "A5_N", "A6_N", "A7_N", "A8_dummy", "A9_N", "A10_N", "A11_dummy", "A12_N", "A13_N", "A14_dummy", "A15_dummy", "A16_N"]]


# In[61]:


data2_model.shape


# In[62]:


x = data2_model[["A1_N", "A2_dummy", "A3_dummy", "A4_N", "A5_N", "A6_N", "A7_N", "A8_dummy", "A9_N", "A10_N", "A11_dummy", "A12_N", "A13_N", "A14_dummy", "A15_dummy"]]


# In[63]:


y = data2_model[["A16_N"]]


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[66]:


xtrain.shape


# In[67]:


ytrain.shape


# In[68]:


xtest.shape


# In[69]:


ytest.shape


# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


logreg = LogisticRegression()


# In[72]:


logreg.fit(xtrain,ytrain)


# In[73]:


preds = logreg.predict(xtest)


# In[74]:


preds


# In[75]:


from sklearn import metrics


# In[76]:


print("Accuracy:", metrics.accuracy_score(ytest,preds))
print("Precision:", metrics.precision_score(ytest,preds))
print("Recall:", metrics.recall_score(ytest,preds))
print("F1 score:", metrics.f1_score(ytest,preds))


# In[77]:


metrics.confusion_matrix(ytest,preds)


# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


dtc = DecisionTreeClassifier()


# In[80]:


dtc.fit(xtrain,ytrain)


# In[81]:


preds = dtc.predict(xtest)


# In[82]:


preds


# In[83]:


print("Accuracy:", metrics.accuracy_score(ytest,preds))
print("Precision:", metrics.precision_score(ytest,preds))
print("Recall:", metrics.recall_score(ytest,preds))
print("F1 score:", metrics.f1_score(ytest,preds))


# In[84]:


x = data2_model[["A2_dummy", "A3_dummy", "A8_dummy", "A11_dummy", "A14_dummy", "A15_dummy"]]


# In[85]:


y = data2_model[["A16_N"]]


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[88]:


xtrain.shape


# In[89]:


ytrain.shape


# In[90]:


xtest.shape


# In[91]:


ytest.shape


# In[92]:


from sklearn.linear_model import LogisticRegression


# In[93]:


logreg = LogisticRegression()


# In[94]:


logreg.fit(x_train,y_train)


# In[95]:


preds = logreg.predict(x_test)


# In[96]:


preds


# In[97]:


print("Accuracy:", metrics.accuracy_score(y_test,preds))
print("Precision:", metrics.precision_score(y_test,preds))
print("Recall:", metrics.recall_score(y_test,preds))
print("F1 score:", metrics.f1_score(y_test,preds))


# In[98]:


metrics.confusion_matrix(y_test,preds)


# In[99]:


x = data2_model[["A1_N", "A4_N", "A5_N", "A6_N", "A7_N", "A9_N", "A10_N", "A12_N", "A13_N"]]


# In[100]:


y = data2_model[["A16_N"]]


# In[101]:


from sklearn.linear_model import LogisticRegression


# In[102]:


logreg = LogisticRegression()


# In[103]:


logreg.fit(x,y)


# In[104]:


from sklearn.model_selection import train_test_split


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[106]:


xtrain.shape


# In[107]:


ytrain.shape


# In[108]:


xtest.shape


# In[109]:


ytest.shape


# In[110]:


preds = logreg.predict(x_test)


# In[111]:


preds


# In[112]:


print("Accuracy:", metrics.accuracy_score(y_test,preds))
print("Precision:", metrics.precision_score(y_test,preds))
print("Recall:", metrics.recall_score(y_test,preds))
print("F1 score:", metrics.f1_score(y_test,preds))


# In[113]:


metrics.confusion_matrix(y_test,preds)


# In[114]:


tp, tn, fp, fn = metrics.confusion_matrix(ytest,preds).ravel()


# In[115]:


print(tp)
print(tn)
print(fp)
print(fn)


# In[116]:


from sklearn.tree import DecisionTreeClassifier


# In[117]:


dtc = DecisionTreeClassifier()


# In[118]:


dtc.fit(x_train,y_train)


# In[119]:


preds = dtc.predict(x_test)


# In[120]:


preds


# In[121]:


print("Accuracy:", metrics.accuracy_score(y_test,preds))
print("Precision:", metrics.precision_score(y_test,preds))
print("Recall:", metrics.recall_score(y_test,preds))
print("F1 score:", metrics.f1_score(y_test,preds))


# In[122]:


tn, fp, fn, tp = metrics.confusion_matrix(y_test,preds).ravel()


# In[123]:


print(tn)
print(fp)
print(fn)
print(tp)


# In[124]:


preds = dtc.predict(x_train)


# In[125]:


preds


# In[126]:


print("Accuracy:", metrics.accuracy_score(y_train,preds))
print("Precision:", metrics.precision_score(y_train,preds))
print("Recall:", metrics.recall_score(y_train,preds))
print("F1 score:", metrics.f1_score(y_train,preds))


# In[127]:


preds = logreg.predict(x_train)


# In[128]:


preds


# In[129]:


print("Accuracy:", metrics.accuracy_score(y_train,preds))
print("Precision:", metrics.precision_score(y_train,preds))
print("Recall:", metrics.recall_score(y_train,preds))
print("F1 score:", metrics.f1_score(y_train,preds))


# In[130]:


logreg.fit(x_train,y_train)


# In[131]:


from sklearn.ensemble import RandomForestClassifier


# In[132]:


rfc = RandomForestClassifier()


# In[133]:


rfc.fit(x_train,y_train)


# In[134]:


rfc_preds = rfc.predict(x_test)


# In[135]:


rfc_preds


# In[136]:


print("Accuracy:", metrics.accuracy_score(y_test,rfc_preds))
print("Precision:", metrics.precision_score(y_test,rfc_preds))
print("Recall:", metrics.recall_score(y_test,rfc_preds))
print("F1 score:", metrics.f1_score(y_test,rfc_preds))


# In[137]:


rfc_preds2 = rfc.predict(x_train)


# In[138]:


rfc_preds2


# In[139]:


print("Accuracy:", metrics.accuracy_score(y_train,rfc_preds2))
print("Precision:", metrics.precision_score(y_train,rfc_preds2))
print("Recall:", metrics.recall_score(y_train,rfc_preds2))
print("F1 score:", metrics.f1_score(y_train,rfc_preds2))


# In[140]:


from sklearn.ensemble import GradientBoostingClassifier


# In[141]:


gbc = GradientBoostingClassifier()


# In[142]:


gbc.fit(x_train,y_train)


# In[143]:


gbc_preds = gbc.predict(x_test)


# In[144]:


gbc_preds


# In[145]:


print("Accuracy:", metrics.accuracy_score(y_test,gbc_preds))
print("Precision:", metrics.precision_score(y_test,gbc_preds))
print("Recall:", metrics.recall_score(y_test,gbc_preds))
print("F1 score:", metrics.f1_score(y_test,gbc_preds))


# In[146]:


gbc_preds = gbc.predict(x_train)


# In[147]:


print("Accuracy:", metrics.accuracy_score(y_train,gbc_preds))
print("Precision:", metrics.precision_score(y_train,gbc_preds))
print("Recall:", metrics.recall_score(y_train,gbc_preds))
print("F1 score:", metrics.f1_score(y_train,gbc_preds))


# In[148]:


from sklearn.preprocessing import StandardScaler


# In[149]:


scaler = StandardScaler()


# In[150]:


scaler.fit(xtrain)


# In[151]:


xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)


# In[152]:


xtrain


# In[153]:


from sklearn.neighbors import KNeighborsClassifier


# In[154]:


classifier = KNeighborsClassifier(n_neighbors=5)


# In[155]:


classifier.fit(xtrain,ytrain)


# In[156]:


preds_k = classifier.predict(xtest)


# In[157]:


preds_k


# In[158]:


print("Accuracy:", metrics.accuracy_score(ytest,preds_k))
print("Precision:", metrics.precision_score(ytest,preds_k))
print("Recall:", metrics.recall_score(ytest,preds_k))
print("F1 score:", metrics.f1_score(ytest,preds_k))


# In[159]:


metrics.confusion_matrix(ytest,preds_k)


# In[ ]:




