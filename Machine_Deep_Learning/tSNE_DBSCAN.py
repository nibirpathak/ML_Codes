#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.spatial import distance
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model
import xlsxwriter 
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


# In[2]:


xls = pd.ExcelFile('Q:/Kirigami/Center/Reduced Data set/coeff.xlsx')
df = pd.read_excel(xls, 'Sheet1')
df.head()


# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1=scaler.fit_transform(df)


for i in range(5):
    a=df1[:,i]
    b=np.reshape(a,(998,1))
    df.iloc[:,[i]]=b
df.head()


# In[5]:


x=df
TN = TSNE(n_components=2, n_iter=3000, perplexity=33).fit_transform(x)
df4=p=pd.DataFrame(data = TN, columns = ['tSNE 1', 'tSNE 2'])
df4.head()


# In[13]:


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=2.55, min_samples=20).fit(df4)
df4['cluster']=clustering.labels_
print(np.unique(clustering.labels_))


# In[14]:


d1 = df4[df4.cluster==-1]
d2 = df4[df4.cluster==0]
d3 = df4[df4.cluster==1]
d4 = df4[df4.cluster==2]
#d5 = df4[df4.cluster==4]
#d6 = df4[df4.cluster==5]
#d7 = df4[df4.cluster==-1]

plt.scatter(d1.iloc[:,0],d1.iloc[:,1],color='red')
plt.scatter(d2.iloc[:,0],d2.iloc[:,1],color='green')
plt.scatter(d3.iloc[:,0],d3.iloc[:,1],color='blue')
plt.scatter(d4.iloc[:,0],d4.iloc[:,1],color='black')
p#lt.scatter(d5.iloc[:,0],d5.iloc[:,1],color='olive')
#plt.scatter(d6.iloc[:,0],d6.iloc[:,1],color='darkviolet')
#plt.scatter(d7.iloc[:,0],d7.iloc[:,1],color='yellow')

plt.title('DBSCAN after t-SNE',fontsize = 12)
plt.xlabel('tSNE 1', fontsize = 12)
plt.ylabel('tSNE 2', fontsize = 12)
fig1=plt.gcf()


# In[119]:


np.size(d3.cluster)


# In[46]:


df4.head(10)
np.size(df4.iloc[:,1])


# In[32]:


# preprocessing for writing the into excel files the model's number belonging to the unique clusters
# index serve as the serial model number 
m=d6.index
m=np.asarray(m)
d6.head(6)


# In[33]:



# writing the into excel files the model's number belonging to the unique clusters
workbook = xlsxwriter.Workbook('Q:/Kirigami/Solar Panel/Center/tSNE_C5.xlsx') 
worksheet = workbook.add_worksheet()
column = 0
row=0
worksheet.write(row, column,'Index') 
row+=1
for item in m : 
        
    worksheet.write(row, column, item) 
  
    # incrementing the value of row by one 
    # with each iteratons. 
    row += 1
      
    
workbook.close()


# In[133]:


itr=3000
perp=21
for i in range(8):
    TN = TSNE(n_components=2, n_iter=itr, perplexity=perp).fit_transform(x)
    df4=pd.DataFrame(data = TN, columns = ['principal component 1', 'principal component 2'])
    
    clustering = DBSCAN(eps=4.5, min_samples=20).fit(df4)
    df4['cluster']=clustering.labels_
    
    d1 = df4[df4.cluster==0]
    d2 = df4[df4.cluster==1]
    d3 = df4[df4.cluster==2]
    d4 = df4[df4.cluster==3]
    d5 = df4[df4.cluster==4]
    d6 = df4[df4.cluster==5]
    d7 = df4[df4.cluster==-1]
    plt.figure(i+1)
    plt.scatter(d1.iloc[:,0],d1.iloc[:,1],color='red')
    plt.scatter(d2.iloc[:,0],d2.iloc[:,1],color='green')
    plt.scatter(d3.iloc[:,0],d3.iloc[:,1],color='blue')
    plt.scatter(d4.iloc[:,0],d4.iloc[:,1],color='black')
    plt.scatter(d5.iloc[:,0],d5.iloc[:,1],color='olive')
    plt.scatter(d6.iloc[:,0],d6.iloc[:,1],color='darkviolet')
    plt.scatter(d7.iloc[:,0],d7.iloc[:,1],color='yellow')

    plt.title('DBSCAN after t-SNE',fontsize = 12)
    plt.xlabel('tSNE 1', fontsize = 12)
    plt.ylabel('tSNE 2', fontsize = 12)

    fig1=plt.gcf()
    fig1.savefig('Q:/Kirigami/Solar Panel/Center/Different_Perplexity/Perp_%s.tif' %perp)
    #itr=itr+250
    perp=perp+1


# In[135]:


df4['cluster'].nunique()


# In[ ]:




