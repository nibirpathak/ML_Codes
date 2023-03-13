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


xls = pd.ExcelFile('Q:/LEPD_24/Cell_type_efficiency.xlsx')
df = pd.read_excel(xls, 'Sheet1')
df.head(8)


# In[15]:


Means1=np.zeros(5)
sems1=np.zeros(5)
k=0
for i in range(0,5):
    Means1[k] = np.mean(df.iloc[:,i])
    sems1[k] = stats.sem(df.iloc[:,i])
    #sems4[i] = np.std(new.iloc[:,i])
    k=k+1
Means1[3]    


# In[30]:


# set width of bar
barWidth = 0.2
 
# set height of bar
bars1 = [Means1[0], Means1[1], Means1[2], Means1[3], Means1[4]]




err1 = [sems1[0], sems1[1], sems1[2], sems1[3], sems1[4]]





r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
print(r1)
print(r2)
r1=[0, 0.3, 0.6, 0.9, 1.2]


# In[33]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(3.0, 2.5), dpi=80, facecolor='w', edgecolor='k')

plt.bar(r1, bars1, color='seagreen', width=barWidth, edgecolor='white')

plt.errorbar( r1, bars1, yerr=err1, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)



# Add xticks on the middle of the group bars
plt.xlabel('Cell Type', fontsize=14, fontname='Arial')
plt.ylabel('Transfection Efficiency(%)', fontsize=14, fontname='Arial')

plt.xticks([r + 0.05*barWidth for r in r1], ['K562', 'NSC', 'IPSC', 'HEK', 'HeLa'],fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim([0,100])
#plt.legend()

plt.savefig("Q:/LEPD_24/Cell_type_efficiency_4.svg", format="svg")


# In[50]:


254

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu1 = 20
variance1 = 10000
sigma1 = math.sqrt(variance)

mu2 = 22
variance2 = 10
sigma2 = math.sqrt(variance)

x1 = np.linspace(mu1 - 0.2*sigma1, mu1 + 0.2*sigma1, 100)
x2 = np.linspace(mu2 - 0.2*sigma2, mu2 + 0.2*sigma2, 100)

plt.plot(x1, stats.norm.pdf(x, mu1, sigma1))
#plt.plot(x2, stats.norm.pdf(x, mu2, sigma2))

plt.xlabel('E*(kT)', fontsize=14, fontname='Arial')
plt.ylabel('Posterior PDF', fontsize=14, fontname='Arial')

plt.show()


# In[ ]:




