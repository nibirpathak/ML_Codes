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
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


# In[2]:


# set width of bar
barwidth = 0.02
 
# set height of bar
bars1 = [28.17]
bars2 = [31.12]
bars3=[61.57]



err1 = [4.07]
err2 = [3.61]
err3=[2.69]







# In[3]:


print(barwidth)


# In[8]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(4.0, 2.5), dpi=80, facecolor='w', edgecolor='k')
plt.bar(0.5*barwidth, bars1, color='seagreen', width=barwidth, edgecolor='white', label='Bulk EP')
plt.bar(2.5*barwidth, bars2, color='firebrick', width=barwidth, edgecolor='white', label='LEPD')
#plt.bar(4.5*barwidth, bars3, color='blue', width=barwidth, edgecolor='white', label='Lipofectamine')



plt.errorbar(0.5*barwidth, bars1, yerr=err1, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)
plt.errorbar( 2.5*barwidth, bars2, yerr=err2, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)
#plt.errorbar( 4.5*barwidth, bars3, yerr=err3, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)


# Add xticks on the middle of the group bars
plt.xlabel('Method', fontsize=14, fontname='Arial')
plt.ylabel('Transfection efficiency (%)', fontsize=14, fontname='Arial')

#xticks=['Bulk EP','LEPD', 'LIPO']
xticks=['Bulk EP','LEPD',]
#x=[0.5*barwidth,2.5*barwidth, 4.5*barwidth]
x=[0.5*barwidth,2.5*barwidth]

plt.xticks(x,xticks,fontsize = 14)

plt.yticks(fontsize = 14)

#plt.legend()

plt.savefig("Q:/LEPD_24/Efficiency_bulk_LEPD_01.svg", format="svg")


# In[64]:


xls = pd.ExcelFile('Q:/Kirigami/Solar Panel/Center/PCA_Kmeans/PCA_C5.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
df1.head()


# In[ ]:





# In[65]:


A=df1.Index
#storing the index(out of 950) of model numbers in A 
A=np.asarray(A)
#getting the number of models in this cluster 
L=len(A)


# In[66]:


xls = pd.ExcelFile('Q:/Kirigami/Solar Panel/Center/tSNE_DBSCAN/Solved_models.xlsx')
df2 = pd.read_excel(xls, 'Sheet1')
df2.head()


# In[67]:


B=df2.Model
#storing the solved model numbers in B 
B=np.asarray(B)
arr=np.zeros(1)
kk=0
models=B[A]
models


# In[68]:


Para=np.zeros((L,10))


# In[69]:


for i in range(0,L):
    #storing the model# in n
    for j in range(0,5000):
        if df.iloc[j,0]== models[i]:
            Para[i,0]=df.iloc[j,1]
            Para[i,1]=df.iloc[j,2]
            Para[i,2]=df.iloc[j,3]
            Para[i,3]=df.iloc[j,4]
            Para[i,4]=df.iloc[j,11]
            Para[i,5]=df.iloc[j,12]
            Para[i,6]=df.iloc[j,13]
            Para[i,7]=df.iloc[j,14]
            Para[i,8]=df.iloc[j,15]
            Para[i,9]=df.iloc[j,16]
            


# In[70]:


df2 = pd.DataFrame(Para)
df2.to_excel(excel_writer = "Q:/Kirigami/Solar Panel/Center/PCA_Kmeans/C5_Param.xlsx")


# In[ ]:




