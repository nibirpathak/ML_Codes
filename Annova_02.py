#!/usr/bin/env python
# coding: utf-8

# In[50]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
import numpy as np


# In[51]:


xls = pd.ExcelFile('Z:/Research\LEPD_24_well\PSED\Device variability_02/Anova_02.xlsx')
df = pd.read_excel(xls, 'Sheet1')
df.head()


# In[52]:


df=df/1e13
df.head()


# In[82]:


df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['D1', 'D2', 'D3', 'D4','D5','D6','D7', 'D8','D9', 'D10', 'D11', 'D12','D13', 'D14', 'D15', 'D16'])
df_melt.columns = ['index', 'Device', 'value']
#value_vars=['D1', 'D2', 'D3', 'D4','D5', 'D6', 'D8','D9', 'D10', 'D11', 'D12','D13', 'D14', 'D16', 'D18','D19', 'D20', 'D21', 'D22','D23', 'D24']
#value_vars=['D1', 'D2', 'D3', 'D4','D5','D6','D7', 'D8','D9', 'D10', 'D11', 'D12','D13', 'D14', 'D15', 'D16']


# In[83]:


df_melt_nozero = df_melt[df_melt.value != 0]
df_melt=df_melt_nozero
#df_melt.to_excel("Z:\Research\LEPD_24_well\PSED\Device variability_02/Data_long.xlsx")


# In[84]:


df_melt.head()


# In[ ]:


#df_melt.to_excel("Z:\Research\LEPD_24_well\PSED\Device variability_02/Data_long.xlsx")


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns

fig1=plt.figure(figsize=(8,4.0))
sns.boxplot(x='Device', y='value', data=df_melt, color='#99c2a2', fliersize=0, linewidth=1.0, width=0.8,)
sns.stripplot(x="Device", y="value", data=df_melt, color='#7d0013',dodge=True,linewidth=0.2,edgecolor='black', jitter=True, s=1)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.ylabel('I(a.u.)', fontsize=12, fontname='Arial')
plt.xlabel('Device', fontsize=12, fontname='Arial')
plt.ylim([0,70])
plt.show()

#fig1.savefig('Z:/Research\LEPD_24_well\PSED\Device variability_02/boxplot.tiff')


# In[ ]:





# In[86]:


# get ANOVA table as R like output
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[87]:


# Ordinary Least Squares (OLS) model
model = ols('value ~ C(Device)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# In[88]:


from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df_melt, res_var='value', anova_model='value ~ C(Device)')
res.anova_summary


# In[89]:


res = stat()
res.tukey_hsd(df=df_melt, res_var='value', xfac_var='Device', anova_model='value ~ C(Device)')
res.tukey_summary


# In[90]:


dfr=res.tukey_summary
dfr[dfr["p-value"] > 0.05]


# In[91]:


# QQ-plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()


# In[92]:


# variance check
#As the data is drawn from normal distribution, use Bartlett’s test to check the Homogeneity of variances. Null hypothesis: samples from populations have equal variances.

#w, pvalue = stats.bartlett(df_melt['D1'], df_melt['D2'], df_melt['D3'], df_melt['D4'],df_melt['D5'], df_melt['D6'], df_melt['D7'], df_melt['D8'], df_melt['D9'], df_melt['D10'], df_melt['D11'], df_melt['D12'],df_melt['D13'], df_melt['D14'], df_melt['D15'], df_melt['D16'])
#print(w, pvalue)


# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the bartlett's test
from bioinfokit.analys import stat 
res = stat()
res.bartlett(df=df_melt, res_var='value', xfac_var='Device')
res.bartlett_summary


# In[93]:


# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the Levene's test
# Levene’s test can be used to check the Homogeneity of variances when the data is not drawn from normal distribution.
from bioinfokit.analys import stat 
res = stat()
res.levene(df=df_melt, res_var='value', xfac_var='Device')
res.levene_summary


# In[ ]:





# In[94]:


# Anova backbone table

data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
anova_table.set_index('Source of Variation', inplace = True)


# In[95]:


# calculate SSTR and update anova table
x_bar = df_melt['value'].mean()
SSTR = df_melt.groupby('Device').count() * (df_melt.groupby('Device').mean() - x_bar)**2
anova_table['SS']['Between Groups'] = SSTR['value'].sum()


# In[96]:


# calculate SSE and update anova table
SSE = (df_melt.groupby('Device').count() - 1) * df_melt.groupby('Device').std()**2
anova_table['SS']['Within Groups'] = SSE['value'].sum()


# In[97]:


# calculate SSTR and update anova table
SSTR = SSTR['value'].sum() + SSE['value'].sum()
anova_table['SS']['Total'] = SSTR


# In[98]:


# update degree of freedom
anova_table['df']['Between Groups'] = df_melt['Device'].nunique() - 1
anova_table['df']['Within Groups'] = df_melt.shape[0] - df_melt['Device'].nunique()
anova_table['df']['Total'] = df_melt.shape[0] - 1


# In[99]:


# calculate MS
anova_table['MS'] = anova_table['SS'] / anova_table['df']

# calculate F 
F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
anova_table['F']['Between Groups'] = F

# p-value
anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

# F critical 
alpha = 0.05
# possible types "right-tailed, left-tailed, two-tailed"
tail_hypothesis_type = "two-tailed"
if tail_hypothesis_type == "two-tailed":
    alpha /= 2
anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

# Final ANOVA Table
anova_table


# In[100]:


# The p-value approach
print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
conclusion = "Failed to reject the null hypothesis."
if anova_table['P-value']['Between Groups'] <= alpha:
    conclusion = "Null Hypothesis is rejected."
print("F-score is:", anova_table['F']['Between Groups'], " and p value is:", anova_table['P-value']['Between Groups'])    
print(conclusion)
    
# The critical value approach
print("\n--------------------------------------------------------------------------------------")
print("Approach 2: The critical value approach to hypothesis testing in the decision rule")
conclusion = "Failed to reject the null hypothesis."
if anova_table['F']['Between Groups'] > anova_table['F crit']['Between Groups']:
    conclusion = "Null Hypothesis is rejected."
print("F-score is:", anova_table['F']['Between Groups'], " and critical value is:", anova_table['F crit']['Between Groups'])
print(conclusion)


# In[101]:


means_02=df_melt.groupby('Device').mean()
means_02


# In[102]:


plt.hist(means_02.value, density=True, bins=10,color='cornflowerblue')  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Mean_Intensity (a.u.)');


# In[103]:


std_01=df_melt.groupby('Device').std()
std_01.head()


# In[104]:


plt.hist(std_01.value, density=True, bins=10, color='skyblue')  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Std_Intensity (a.u.)')


# In[105]:


plt.hist((std_01.value/means_02.value)*100, density=True, bins=10, color='dodgerblue')  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('COV_data %')


# In[106]:


plt.hist(df_melt.value, bins='auto', histtype='bar', ec='k') 
plt.ylabel('Probability')
plt.xlabel('Intensity (a.u.)')
plt.xlim([10,70])


# In[ ]:





# In[108]:


xls = pd.ExcelFile('Z:/Research\LEPD_24_well\PSED\Device variability_02/All_means.xlsx')
df_m = pd.read_excel(xls, 'Sheet2')


# In[109]:


df_m


# In[110]:


uniform_data=df_m.iloc[0:4,0:4]


# In[113]:


uniform_data=np.asarray(uniform_data)


# In[121]:


import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
# creating a colormap
colormap = sns.color_palette("Greens")
ax = sns.heatmap(uniform_data, linewidth=0.5, cmap=colormap)
plt.show()


# In[ ]:




