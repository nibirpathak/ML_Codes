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


xls = pd.ExcelFile('D:/20000 data sets/Coeff.xlsx')
df = pd.read_excel(xls, 'Sheet1')
df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1=scaler.fit_transform(df)


# In[4]:


for i in range(5):
    a=df1[:,i]
    b=np.reshape(a,(19210,1)) #19210, 13497
    df.iloc[:,[i]]=b
df.head()


# In[5]:


x=df
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
PC = pca.fit_transform(x)
pca.explained_variance_ratio_


# In[6]:


df3=p=pd.DataFrame(data = PC, columns = ['principal component 1', 'principal component 2'])
df3.head()


# In[7]:


from sklearn.cluster import KMeans
sse = []
k_rng = range(1,15)
for k in k_rng:
    km = KMeans(n_clusters=k,init='random',algorithm='auto')
    km.fit(df3)
    #km.fit(df)
    cluster_labels = km.fit_predict(df3)
    #cluster_labels = km.fit_predict(df)
    sse.append(km.inertia_)
    #ssc.append(silhouette_score(df3, cluster_labels))


# In[8]:


#plt.xlabel('K',fontsize = 15)
#plt.ylabel('Sum of squared error',fontsize = 15)
plt.figure(figsize=(2.0,1.5))
plt.figure(1)
plt.plot(k_rng,sse,'ro-', ms=1)
#print(ssc)
#fig1=plt.gcf()
plt.xlabel('# clusters (k)', fontsize=10, fontname='Arial')
plt.ylabel('SSE', fontsize=10, fontname='Arial')
plt.xticks(np.arange(0, 15, 2),fontsize = 8,fontname='Arial')
plt.yticks(fontsize = 8, fontname='Arial')
#fig1.savefig('D:/20000 data sets/elbowplot.svg', format="svg")


# In[9]:


km = KMeans(n_clusters=5, init= 'random', n_init = 10, algorithm='auto', tol= 1e-6, max_iter=300, random_state = None)
#km = KMeans(n_clusters=3, init= 'k-means++', n_init = 10, algorithm='full', tol= 1e-6, max_iter=300, random_state = None)
y_predicted = km.fit_predict(df3)
#y_predicted = km.fit_predict(df)


# In[10]:


df3.head()


# In[11]:


df3['cluster']=y_predicted
df3.head(6)


# In[12]:


d1 = df3[df3.cluster==0]
d2 = df3[df3.cluster==1]
d3 = df3[df3.cluster==2]
d4 = df3[df3.cluster==3]
d5 = df3[df3.cluster==4]


# In[13]:


plt.figure(figsize=(2.0,1.5))
plt.figure(1)
plt.scatter(d1.iloc[:,0],d1.iloc[:,1],color='red', s=0.25)
plt.scatter(d2.iloc[:,0],d2.iloc[:,1],color='green', s=0.25)
plt.scatter(d3.iloc[:,0],d3.iloc[:,1],color='blue', s=0.25)
plt.scatter(d4.iloc[:,0],d4.iloc[:,1],color='black',s=0.25)
plt.scatter(d5.iloc[:,0],d5.iloc[:,1],color='orange',s=0.25)
plt.title('K-Means after PCA',fontsize = 12)
plt.xlabel('Principal Component 1', fontsize = 10)
plt.ylabel('Principal Component 2', fontsize = 10)
fig1=plt.gcf()
#plt.savefig('D:/20000 data sets/PCA_Clusters_5.svg', format="svg")


# In[21]:


# preprocessing for writing the into excel files the model's number belonging to the unique clusters
# index serve as the serial model number 
m=d3.index
m=np.asarray(m)
print(np.size(m))
d3.head(6)


# In[22]:


# writing the into excel files the model's number belonging to the unique clusters
workbook = xlsxwriter.Workbook('D:/Data set with proper imperfection/PCA_C3.xlsx') 
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


# In[23]:


df3.to_excel("D:/Data set with proper imperfection/output_3Cs.xlsx")


# In[ ]:




