#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import *
import pylab


# In[2]:


df = pd.read_csv("Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 2-BSA delivery in immortalized cells/Figure 2d-Voltage Variation/Intensity_all.csv")
df1 = pd.read_csv("Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 2-BSA delivery in immortalized cells/Figure 2d-Voltage Variation/Avg_BSA_Hela.csv")
df0 = pd.read_csv("Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 2-BSA delivery in immortalized cells/Figure 2d-Voltage Variation/Intensity_Voltage_BSA_Hela.csv")
colors1 = sns.cubehelix_palette(8, start=.5, rot=-.75)
color1 = sns.set_palette(sns.color_palette(colors1))
df
#Intensity_Voltage_BSA_Hela


# In[12]:


fig, ax = plt.subplots(figsize=(2.5, 2.6))

bar = sns.barplot(y='Intensity', x='Voltage', data=df, estimator=mean, palette=color1, linewidth=1.5, ci=68, ax=ax, capsize=0.1)
for c in bar.patches:
    c.set_zorder(0)

sns.regplot(x=np.arange(0, len(df)), y='Means', data=df, order=2,  line_kws={'linestyle':'--','linewidth': 1, 'color': 'black'}, ci=None, truncate=True, ax=ax)

'''
y1=df.Means
x1=df.Potential
z = np.polyfit(x1[0:4], y1[0:4], 2)
xp = np.linspace(10, 20, 100)
p = np.poly1d(z)
plt.plot(xp, p(xp),'-')
'''
ax.set_xlabel('Voltage (V)', fontsize=16, fontname='Arial')
ax.set_ylabel('I (a.u.)', fontsize=16, fontname='Arial')
#ax.set_xlabel('Voltage(V)',fontsize=14,fontname='Arial')
#ax.set_ylabel('Fluorescence Intensity(a.u.)',fontsize=14,fontname='Arial')
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)

fig1=plt.gcf()
fig1.savefig('Q:/Small_response/Voltage_var.svg', format='svg')


# In[23]:


V=np.array([0, 10, 12, 15 ,18 ,20])
I=np.array([0, 250.28, 320.07, 376.88, 467.49, 712.29])
yerr=np.array([0, 19.37, 29.50, 38.03, 42.59, 69.80])


# In[24]:


q = 2
zq = polyfit(V,I,q) 
pq = poly1d(zq)


# In[25]:


xx = linspace(0, max(V), 500)
fig, ax = plt.subplots(figsize=(2.5, 2.6))
pylab.plot(xx, pq(xx),'-g')
pylab.errorbar(V, I,yerr=yerr, fmt='r.')
ax.set_xlabel('Voltage (V)', fontsize=16, fontname='Arial')
ax.set_ylabel('I (a.u.)', fontsize=16, fontname='Arial')
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)

#plt.savefig('Z:/Papers/NFP-E Delivery and  Sampling/ACS Nano/Figures/Figure 2-BSA delivery in immortalized cells/Figure 2d-Voltage Variation/Fig2g.svg') 


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * (x-10) ** 2 + b*(x-10)+c

V=np.array([10, 12, 15 ,18 ,20])
I=np.array([250.28, 320.07, 376.88, 467.49, 712.29])

print(np.polyfit(V, I, 2))

popt, _ = curve_fit(func, V, I)
print(popt)


xx = linspace(10, max(V), 500)

plt.plot(V, I, 'bo')
plt.plot(xx, func(xx, *popt), 'k-')

plt.show()


# In[ ]:




