#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # 3 데이터셋 로드

# In[ ]:


# 기본적인 통계, 분포, 결측치


# In[2]:


df = pd.read_csv('data/mental_health_data.csv')
df.shape 


# In[6]:


df.head()


# In[5]:


df.info()


# In[7]:


# 결측치

df_null=df.isnull()
df_null.head()
# 깨끗


# In[8]:


df.describe() 


# # 4 학습 예측 데이터셋 만들기

# In[13]:


# X : feature  y : label값
df.columns


# In[16]:


X = df[['YearStart', 'YearEnd', 'LocationAbbr', 'LocationDesc', 'DataSource',
       'Topic', 'Question', 'DataValueUnit', 'DataValueType', 'DataValue',
       'DataValueAlt', 'DatavalueFootnote', 'LowConfidenceLimit', 'StratificationCategory1', 'Stratification1',
       'GeoLocation', 'LocationID', 'TopicID', 'QuestionID', 'DataValueTypeID',
       'StratificationCategoryID1', 'StratificationID1']]
X.shape


# In[ ]:


## 'HighConfidenceLimit'


# In[18]:


y=df['HighConfidenceLimit']
y.shape


# In[19]:


# 사이킷런에서 제공하는 model_selection 의 train_test_split으로 만든다.

from sklearn.model_selection import train_test_split

# train_test_split?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[22]:


# train 세트의 문제와 정답의 데이터 수를 확인

X_train
# random_state=42 를 지우면 매번 랜덤이 되어서 똑같은 자료를 불러올 수 없으므로 지정


# In[23]:


X_train.shape, y_train.shape


# In[24]:


# test 세트의 문제와 정답의 데이터 수 확인
X_test.shape, y_test.shape


# # 5 GradientBoost

# In[27]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 6 학습, 예측

# In[28]:


# 학습

model.fit(X_train, y_train)


# In[ ]:




