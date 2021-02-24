#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import tensorflow as tf


# ## 数据清洗

# In[66]:


#读取评分表
ratings_df = pd.read_csv('ml-latest-small\\ratings.csv')
ratings_df.tail()


# In[67]:


#读取电影表
movies_df = pd.read_csv('ml-latest-small\\movies.csv')
movies_df.tail()


# In[68]:


#加入索引行
movies_df['MovieRow'] = movies_df.index
movies_df.tail()


# ## 特征提取

# In[69]:


#提取索引、电影id和标题
movies_df = movies_df[['MovieRow', 'movieId', 'title']]
movies_df.tail()


# In[70]:


#暂存到文件
movies_df.to_csv('ml-latest-small\\moviesProcessed.csv', index = False, header = True, encoding = 'utf-8')


# In[71]:


#合并两表
ratings_df = pd.merge(ratings_df, movies_df, on = 'movieId')
ratings_df.head()


# In[72]:


#特征提取
ratings_df = ratings_df[['userId', 'MovieRow', 'rating']]
ratings_df.head()


# ## 创建评分矩阵rating和评分记录矩阵record

# In[73]:


#获取评分的用户和电影的数量
userNo = ratings_df['userId'].max() + 1
movieNo = ratings_df['MovieRow'].max() + 1

rating = np.zeros((movieNo, userNo))

#遍历矩阵，将评分填入表中
for  index, row in ratings_df.iterrows():
    rating[int(row['MovieRow']), int(row['userId'])] = row['rating']

rating


# In[74]:


#record记录用户是否评分，1表示评分，0表示没有
record = rating > 0
record = np.array(record, dtype = int)
record


# ## 构建模型

# In[75]:


#标准化评分
def normalizeRatings(rating, record):
    #获取电影数量m和用户数量n
    m, n = rating.shape
    #rating_mean 电影平均分
    #rating_norm 标准化后的电影评分
    rating_mean = np.zeros((m, 1))
    rating_norm = np.zeros((m, n))
    for i in range(m):
        index = record[i, :] != 0
        rating_mean[i] = np.mean(rating[i, index])
        rating_norm[i, index] -= rating_mean[i]
    #将nan转成数字0
    rating_norm = np.nan_to_num(rating_norm)
    rating_mean = np.nan_to_num(rating_mean)
    return rating_norm, rating_mean


# ## 损失函数
# ## $J(\theta)=\dfrac{1}{2}\sum_{j=1}^{u}\sum_{i,r(i,j)=1}{}((\theta^{j})^{T}x^{i}-y^{(i,j)})^2+\dfrac{\lambda}{2}\sum_{j=1}^{u}\sum_{k=1}^{n}(\theta_{k}^{j})^2$
# ### 其中，x表示电影的标签，大小1\*n，$\theta$表示用户j的喜好，大小也是1\*n，这里的n就是tags的数量。y(i,j)表示用户j对电影i的评分，u是用户数量，后面一部分是正则化项，防止过拟合。

# In[76]:


#设置参数，其中X_parameters对应x^i，Theta_parameters对应\theta^j
num_features = 10
X_parameters = tf.Variable(tf.random.normal([movieNo, num_features], stddev = 0.35))
Theta_parameters = tf.Variable(tf.random.normal([userNo, num_features], stddev = 0.35))


# In[78]:


rating_norm, rating_mean = normalizeRatings(rating, record)
#损失函数
loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b=True) - rating_norm) * record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))
#Adam算法优化器，设置学习率为0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
var_list = [X_parameters, Theta_parameters]
train = optimizer.minimize(loss, var_list)
tf.summary.scalar('loss', loss)


# In[ ]:




