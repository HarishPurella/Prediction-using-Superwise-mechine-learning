#!/usr/bin/env python
# coding: utf-8

# ##                                 Prediction using Superwise mechine learning
# 
# 
# ##                                                GRIP:The Spark Foundation 
# 
# ##                                 Data science &Business Analytics Intern(GRIPMAY21)

# ## Task#1

# In[ ]:


#Predict the percentage of on student based on the no.of study hours.


# #1.load the labraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[91]:


#Read the data with pandas


# In[ ]:


newdata=pd.read_csv("http://bit.ly/w-data")


# In[ ]:


#Analize data set


# In[132]:


newdata.info


# In[97]:


newdata.head()


# In[98]:


newdata.shape


# In[99]:


#describe the dataset


# In[100]:


newdata.describe()


# In[101]:


newdata.isnull().sum()


# In[102]:


#visualization the data to better understanding


# In[103]:


newdata.plot(x="Hours",y="Scores",style="o")
plt.title("Study hours and scores")
plt.xlabel("no.of Hours")
plt.ylabel("percentage Scores")


# In[104]:


#Divided the data into input and out variable


# In[105]:


x=newdata.iloc[:, :-1].values


# In[106]:


y=newdata.iloc[:, 1].values


# In[107]:


#Split the data into train and test set


# In[108]:


from sklearn.model_selection import train_test_split


# In[109]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=0)


# In[110]:


x_train.shape,y_train.shape


# In[111]:


y_test.shape,x_test.shape


# In[112]:


#Apply the linear regression on train dataset


# In[113]:


from sklearn.linear_model import LinearRegression


# In[114]:


lm=LinearRegression()


# In[115]:


lm.fit(x_train,y_train)


# In[116]:


# To Get parameters


# In[117]:


lm.intercept_


# In[118]:


lm.coef_


# In[119]:


line=lm.coef_*x+lm.intercept_


# In[120]:


plt.scatter(x,y,color='red')
plt.plot(x,line)
plt.title("Study hours and scores")
plt.xlabel("no.of Hours")
plt.ylabel("percentage Scores")
plt.grid()
plt.show()


# In[121]:


#predictions on the set dataset


# In[122]:


print(y_test)
print(x_test)
y_pred=lm.predict(x_test)


# In[123]:


df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df


# In[124]:


#predicting the newdata


# In[125]:


hours=np.array([9.30]).reshape(1,-1)
hours=lm.predict(hours)
hours


# In[126]:


Hours=float(input("entera value: "))


# In[127]:


new_pred=lm.predict([[Hours]])
print("no of hours:{}".format(Hours))
print("predicted score:{}".format(new_pred))


# In[ ]:





# In[128]:


#evaluations


# In[129]:


from sklearn import metrics
print("mean absolute error:",metrics.mean_absolute_error(y_test,y_pred))


# In[130]:


#calculating mean square error


# In[131]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# In[84]:


#These steps will show how we evaluate the linear regression model.


# In[85]:


#Thanq


# In[ ]:




