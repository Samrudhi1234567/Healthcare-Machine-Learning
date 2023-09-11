#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"C:\Users\SHREE\Desktop\Samrudhi\sam\petrol consumption on ML model.csv")


# In[3]:


data.head(10)


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.plot (kind = "box")


# In[8]:


data["Petrol tax (cents per gallon)"].plot (kind = "box")


# In[9]:


q3 = data["Petrol tax (cents per gallon)"].quantile(0.75)
q3


# In[10]:


q1 = data["Petrol tax (cents per gallon)"].quantile(0.25)
q1


# In[11]:


iqr_petrol_tax = q3-q1
iqr_petrol_tax


# In[12]:


Uledge_petrol_tax = q3 + (1.5*iqr_petrol_tax)
Uledge_petrol_tax


# In[13]:


petrol_tax_dummy = []
for i in data["Petrol tax (cents per gallon)"]:
    if i > 9.8125:
        petrol_tax_dummy.append(9.8125)
    else:
        petrol_tax_dummy.append(i)


# In[14]:


data["petrol_tax_dummy"] = petrol_tax_dummy


# In[15]:


Lledge_petrol_tax = q1 - (1.5*iqr_petrol_tax)
Lledge_petrol_tax


# In[16]:


petrol_tax_dummy1 = []
for i in data["petrol_tax_dummy"]:
    if i < 5.3125:
        petrol_tax_dummy1.append(5.3125)
    else:
        petrol_tax_dummy1.append(i)


# In[17]:


data["petrol_tax_dummy1"] = petrol_tax_dummy1


# In[18]:


data["petrol_tax_dummy1"].plot (kind = "box")


# In[19]:


data["Average income (dollars)"].plot (kind = "box")


# In[20]:


data["Paved Highways (miles)"].plot (kind = "box")


# In[21]:


q3 = data["Paved Highways (miles)"].quantile(0.75)
q3


# In[22]:


q1 = data["Paved Highways (miles)"].quantile(0.25)
q1


# In[23]:


iqr_paved_highway = q3-q1
iqr_paved_highway


# In[24]:


Uledge_paved_highway = q3 + (1.5*iqr_paved_highway)
Uledge_paved_highway


# In[25]:


paved_highway_dummy = []
for i in data["Paved Highways (miles)"]:
    if i > 13224.625:
        paved_highway_dummy.append(13224.625)
    else:
        paved_highway_dummy.append(i)


# In[26]:


data["paved_highway_dummy"] = paved_highway_dummy


# In[27]:


data["paved_highway_dummy"].plot (kind = "box")


# In[28]:


data["Population_Driver_licence(%)"].plot (kind = "box")


# In[29]:


q3 = data["Population_Driver_licence(%)"].quantile(0.75)
q3


# In[30]:


q1 = data["Population_Driver_licence(%)"].quantile(0.25)
q1


# In[31]:


iqr_dr_license = q3-q1
iqr_dr_license


# In[32]:


Uledge_dr_license = q3 + (1.5*iqr_dr_license)
Uledge_dr_license


# In[33]:


dr_license_dummy = []
for i in data["Population_Driver_licence(%)"]:
    if i > 0.6934:
        dr_license_dummy.append(0.6934)
    else:
        dr_license_dummy.append(i)


# In[34]:


data["dr_license_dummy"] = dr_license_dummy


# In[35]:


data["dr_license_dummy"].plot (kind = "box")


# In[36]:


data["Consumption of petrol (millions of gallons)"].plot (kind = "box")


# In[37]:


q3 = data["Consumption of petrol (millions of gallons)"].quantile(0.75)
q3


# In[38]:


q1 = data["Consumption of petrol (millions of gallons)"].quantile(0.25)
q1


# In[39]:


iqr_petrol_consmp = q3-q1
iqr_petrol_consmp


# In[40]:


Uledge_petrol_consmp = q3 + (1.5*iqr_petrol_consmp)
Uledge_petrol_consmp


# In[41]:


petrol_consmp_dummy = []
for i in data["Consumption of petrol (millions of gallons)"]:
    if i > 817.625:
        petrol_consmp_dummy.append(817.625)
    else:
        petrol_consmp_dummy.append(i)


# In[42]:


data["petrol_consmp_dummy"] = petrol_consmp_dummy


# In[43]:


data["petrol_consmp_dummy"].plot (kind = "box")


# In[44]:


data.plot (kind = "box")


# In[45]:


data.info()


# In[46]:


data.dtypes


# In[47]:


data_model = data[["petrol_tax_dummy1", "Average income (dollars)", "paved_highway_dummy", "dr_license_dummy", "petrol_consmp_dummy"]]


# In[48]:


data_model.info()


# In[49]:


data_model.describe()


# In[50]:


data_model.plot (kind = "box")


# In[51]:


x = data_model[["petrol_tax_dummy1", "Average income (dollars)", "paved_highway_dummy", "dr_license_dummy"]]


# In[52]:


y = data_model[["petrol_consmp_dummy"]]


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[55]:


x_train.shape


# In[56]:


y_train.shape


# In[57]:


x_test.shape


# In[58]:


y_test.shape


# In[59]:


from sklearn.linear_model import LinearRegression


# In[60]:


reg = LinearRegression()


# In[61]:


reg.fit(x_train,y_train)


# In[62]:


reg.coef_


# In[63]:


reg.intercept_


# In[64]:


reg.score


# In[65]:


preds = reg.predict(x_test)


# In[66]:


preds


# In[67]:


from sklearn import metrics


# In[68]:


print("mean absolute error:", metrics.mean_absolute_error(y_test,preds))
print("mean squared error:", metrics.mean_squared_error(y_test,preds))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,preds)))
print("r square:", metrics.r2_score(y_test,preds))


# In[69]:


from sklearn.tree import DecisionTreeRegressor


# In[70]:


dtc = DecisionTreeRegressor()


# In[71]:


dtc.fit(x_train,y_train)


# In[72]:


preds_dtc = dtc.predict(x_test)


# In[73]:


preds_dtc


# In[74]:


print("mean absolute error:", metrics.mean_absolute_error(y_test,preds_dtc))
print("mean squared error:", metrics.mean_squared_error(y_test,preds_dtc))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,preds_dtc)))
print("r square:", metrics.r2_score(y_test,preds_dtc))


# In[75]:


from sklearn.ensemble import RandomForestRegressor


# In[76]:


rfc = RandomForestRegressor()


# In[77]:


rfc.fit(x_train,y_train)


# In[78]:


preds_rfc = rfc.predict(x_test)


# In[79]:


preds_rfc


# In[80]:


print("mean absolute error:", metrics.mean_absolute_error(y_test,preds_rfc))
print("mean squared error:", metrics.mean_squared_error(y_test,preds_rfc))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,preds_rfc)))
print("r square:", metrics.r2_score(y_test,preds_rfc))


# In[81]:


from sklearn.ensemble import GradientBoostingRegressor


# In[82]:


gbr = GradientBoostingRegressor()


# In[83]:


gbr.fit(x_train,y_train)


# In[84]:


preds_gbr = gbr.predict(x_test)


# In[85]:


preds_gbr


# In[86]:


print("mean absolute error:", metrics.mean_absolute_error(y_test,preds_gbr))
print("mean squared error:", metrics.mean_squared_error(y_test,preds_gbr))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,preds_gbr)))
print("r square:", metrics.r2_score(y_test,preds_gbr))


# In[87]:


from sklearn.neighbors import KNeighborsRegressor


# In[88]:


knn = KNeighborsRegressor()


# In[89]:


knn.fit(x_train,y_train)


# In[90]:


preds_knn = knn.predict(x_test)


# In[91]:


preds_knn


# In[92]:


print("mean absolute error:", metrics.mean_absolute_error(y_test,preds_knn))
print("mean squared error:", metrics.mean_squared_error(y_test,preds_knn))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,preds_knn)))
print("r square:", metrics.r2_score(y_test,preds_knn))


# The Linear Regression Model is performing well for this data set as the prediction error rate is less and the r square value is more as compared to other models.
