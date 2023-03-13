#!/usr/bin/env python
# coding: utf-8

# In[6]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
import numpy as np
import xlsxwriter 


# In[3]:


df = pd.read_csv("Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figures 5,6-Ratiometric Transfection/Ratiometric.csv")
df.head()


# In[4]:


colors = sns.diverging_palette(160, 8, l=60, sep=20, n=3, center="dark")
color1 = sns.set_palette(sns.color_palette(colors))

plt.figure(figsize=(3.6, 3.6))
bplot = sns.scatterplot(data=df,x='mCh', y='EGFP', hue='EGFP:mCh', palette=color1)

plt.yticks(np.arange(0, 110, 20))
plt.xticks(np.arange(0, 110, 20))
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlabel('I(mCh)', fontsize=16, fontname='Arial')
plt.ylabel('I(EGFP)', fontsize=16, fontname='Arial')
plt.legend(fontsize='13', title_fontsize='16', loc='upper right')

plt.show()


# In[5]:


from scipy.stats.stats import pearsonr


# In[7]:


xls = pd.ExcelFile('Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figures 5,6-Ratiometric Transfection/Pearson_ratiometric.xlsx')
df = pd.read_excel(xls, 'Sheet1')
df.head(10)


# In[12]:


a=df.EGFP1
b=df.mCh1

c=df.EGFP2
d=df.mCh2

e=df.EGFP3
f=df.mCh3

a=np.asarray(a)
b=np.asarray(b)
c=np.asarray(c)
d=np.asarray(d)
e=np.asarray(e)
f=np.asarray(f)

nan_array = np.isnan(a)
not_nan_array = ~ nan_array
a1 = a[not_nan_array]

nan_array = np.isnan(b)
not_nan_array = ~ nan_array
b1 = b[not_nan_array]

nan_array = np.isnan(c)
not_nan_array = ~ nan_array
c1 = c[not_nan_array]

nan_array = np.isnan(d)
not_nan_array = ~ nan_array
d1 = d[not_nan_array]

nan_array = np.isnan(e)
not_nan_array = ~ nan_array
e1 = e[not_nan_array]

nan_array = np.isnan(f)
not_nan_array = ~ nan_array
f1 = f[not_nan_array]



# In[25]:


C1,P1 = pearsonr(a1, b1)
C2,P2 = pearsonr(c1, d1)
C3,P3 = pearsonr(e1, f1)


# In[26]:


print('For 20:1, the pearson coorelation coeffficient is %s' %(C1))
print('For 1:1, the pearson coorelation coeffficient is %s' %(C2))
print('For 1:20, the pearson coorelation coeffficient is %s' %(C3))


# In[42]:


plt.hist(e1, bins=30)
plt.show()


# In[ ]:




