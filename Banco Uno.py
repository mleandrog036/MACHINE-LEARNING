#!/usr/bin/env python
# coding: utf-8

# <h2 Style="color:#1E90FF">Cargando y Examinando los datos</h2>

# <h2 Style="color:#1E90FF">Conociendo los datos y su dominio</h2>
# 
# 

# In[3]:


from sqlalchemy import create_engine
import pymysql
import pandas as pd


# In[ ]:


connection = pymysql.connect(host='data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com',
user='deepAnalytics',
password='Sqltask1234!',
database='Credit',
charset='utf8mb4',
cursorclass=pymysql.cursors.DictCursor)


# In[4]:


df = pd.read_csv("BancoUno.csv")


# In[5]:


df.head(10)


# In[6]:


df.shape


# In[7]:


df.tail(10)


# <h2 Style="color:#1E90FF">Preparando los Datos Importados</h2>
# 
# <center><h6>Paso 1: Cambiando el nombre al DataFrame</h6></center>

# In[8]:


credit = df


# In[9]:


credit.head()


# <center><h6>Paso 2: Aplicando Pandas Profiling</h6></center>

# In[10]:


import pandas_profiling


# In[11]:


pandas_profiling.ProfileReport(credit)


# In[12]:


credit["PAY_2"].value_counts()


# In[13]:


credit.index[credit["PAY_2"]=="PAY_2"].tolist()


# In[14]:


credit = credit.drop(credit.index[2397])


# In[15]:


print(credit[["PAY_2"]].to_string(index=2397))


# In[17]:


credit.shape


# In[18]:


credit.duplicated()


# In[16]:


credit = credit.drop_duplicates()


# In[17]:


credit.duplicated()


# In[21]:


credit.drop(index= 0, axis = 0)


# In[18]:


credit.isnull().sum()


# <center><h6>Paso 3: Trabajando con datos no numéricos (One-Hot Encoding)</h6></center>

# In[19]:


credit.dtypes


# In[20]:


credit [["LIMIT_BAL","AGE","MARRIAGE", "PAY_0","PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = credit [["LIMIT_BAL","AGE","MARRIAGE", "PAY_0","PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] .astype ("int")


# In[21]:


credit [["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5","BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]] = credit [["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5","BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]] .astype ("float")


# In[22]:


credit.dtypes


# In[23]:


credit = pd.get_dummies(credit)


# In[24]:


pandas_profiling.ProfileReport(credit)


# <h2 Style="color:#1E90FF">Visualizando los datos.</h2>
# 

# <center><h3>Visualización de Datos</h3></center>
# </br>
# <p Style="text-aling: justify" >Histogramas</p>

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt


# In[27]:


header = credit.dtypes.index
print(header)

#para revisar los nombres 


# In[28]:


plt.hist(credit['LIMIT_BAL'])
plt.show()


# In[29]:


plt.hist(credit['LIMIT_BAL'], bins=4)

#verificar número de bins


# <p Style="text-aling: justify" >Gráficas de Líneas</p>

# In[30]:


plt.plot(credit['LIMIT_BAL'])
plt.show()


# <p Style="text-aling: justify" >Gráficas de Dispersión</p>

# In[31]:


x = credit['PAY_0']

y = credit['PAY_2']

plt.scatter(x,y)
plt.show()


# <h2 Style="color:#1E90FF">Correlación</h2>
# 

# In[32]:


corrMat = credit.corr()
print(corrMat)


# <h2 Style="color:#1E90FF">Covarianza</h2>

# In[33]:


covMat = credit.cov()
print(covMat)


# In[34]:


plt.scatter(credit.BILL_AMT1, credit.PAY_AMT1, color="blue")
plt.xlabel("BILL_AMT1")
plt.ylabel("PAY_AMT1")
plt.show()


# In[35]:


plt.scatter(credit.LIMIT_BAL, credit.BILL_AMT1, color="blue")
plt.xlabel("LIMIT_BAL")
plt.ylabel("BILL_AMT1")
plt.show()


# In[54]:


credit.head(5)


# In[36]:



credit = df[['LIMIT_BAL','MARRIAGE','AGE']]
credit.head(9)


# In[47]:


import numpy as np 
import pandas as pd 
import scipy
from math import sqrt 
import matplotlib. pyplot as pit


# In[53]:


from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
from sklearn import linear_model


# In[55]:


from sklearn .metrics import mean_squared_error 
from sklearn.metrics import r2_score 
from sklearn.model_selection import cross_val_score


# In[67]:


#from sklearn.cross_validation import train_test_split


# In[ ]:




