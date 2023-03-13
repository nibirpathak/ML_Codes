#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean


# In[4]:


df = pd.read_csv("Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 4-Plasmid Transfection/NFP_LIPO_violin.csv")
df.head()


colors = sns.color_palette("Greens_r")
color1 = sns.set_palette(sns.color_palette(colors))
plt.figure(figsize=(2.5,2.6))

bplot=sns.violinplot(x="Method", y="Intensity", data=df, palette=color1)

plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.xlabel('Method', fontsize=16, fontname='Arial')
plt.ylabel('I(a.u.)', fontsize=16, fontname='Arial')
#plt.ylim(-20,150)
#fig1=plt.gcf()
#fig1.savefig('Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 4-Plasmid Transfection/Figure_3f1.svg')
plt.show()


# In[3]:


sns.__version__


# In[ ]:




