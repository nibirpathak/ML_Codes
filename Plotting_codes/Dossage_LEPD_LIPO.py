#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy import stats


# In[3]:


'''
xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Ratiometric/Dossage spread/Dossage.xlsx')
df1 = pd.read_excel(xls, 'Combined')
df1.head(6)
'''


# In[66]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1_sc=scaler.fit_transform(df1.iloc[:,0:4])


# In[67]:


columns=['LEPD(GFP)', 'LEPD(mch)', 'LIPO(GFP)', 'LIPO(mch)']
df1_sc=pd.DataFrame(df1_sc, columns=columns)
df1_sc.head()


# In[68]:


df1_sc.to_excel("Z:/Papers/LEPD_24_well_plate/Ratiometric/Dossage spread/Dossage_stand.xlsx") 


# In[69]:


xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Ratiometric/Dossage spread/Dossage_stand.xlsx')
df2 = pd.read_excel(xls, 'Sheet1')
df2.head(6)


# In[48]:


# set width of bar
barwidth = 0.03
 
# set height of bar
gfp = [0.23, 0.14]
mch = [0.21, 0.07]

err1 = [0.03, 0.03]
err2 = [0.02, 0.02]

r1 = [0.1, 0.2]
r2 = [x + barwidth for x in r1]
print(r1)


# In[60]:


plt.bar(r1, gfp, color='seagreen', width=barwidth, edgecolor='white', label='GFP')
plt.bar(r2, mch, color='firebrick', width=barwidth, edgecolor='white', label='mcherry')

plt.errorbar(r1, gfp, yerr=err1, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)
plt.errorbar(r2, mch, yerr=err2, fmt='o', markersize='2', color='Black', elinewidth=1.5,capthick=0.5, capsize=2)

# Add xticks on the middle of the group bars
plt.xlabel('Method', fontsize=14, fontname='Arial')
plt.ylabel('Normalized Intensity (a.u.)', fontsize=14, fontname='Arial')

xticks=['LEPD', 'LIPO']
x=[r1[0]+0.5*barwidth,r1[1]+0.5*barwidth]
plt.xticks(x,xticks,fontsize = 14)

plt.yticks(fontsize = 14)

plt.legend()

plt.savefig("Z:/Papers/LEPD_24_well_plate/Ratiometric/Dossage spread/Dossage_bar.png", format="png")


# In[2]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Ratiometric/Dossage spread/Box.xlsx')
df3 = pd.read_excel(xls, 'All')
df3.head(6)


# In[17]:


colors = sns.color_palette("Greens_r")
color1 = sns.set_palette(sns.color_palette(colors))
plt.figure(figsize=(2.8,3.0))

bplot=sns.violinplot(x="Method", y="Norm_data", data=df3, palette=color1)

plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlabel('Method', fontsize=16, fontname='Arial')
plt.ylabel('I(a.u.)', fontsize=16, fontname='Arial')
#plt.ylim(-20,150)
plt.show()


# In[32]:


# Load data set with GFP data for Hela cells transfected via both LEPD and LIPO
xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa/Data_1.xlsx')
df4 = pd.read_excel(xls, 'HeLa')
df4.head(6)


# In[34]:


# Plot distribution of Normalized intensities 
plt.figure(figsize=(4.0,2.5))
color2=['cornflowerblue','salmon']

sns.kdeplot(data=df4, x="Norm_all", hue="Method", palette=color2, multiple="layer", log_scale=None, thresh=0.9, fill=True,bw_adjust = 0.3)

plt.yticks(fontsize = 8)
plt.xticks(fontsize = 8)
plt.xlabel('Intensity(a.u.)', fontsize=10, fontname='Arial')
plt.ylabel('Density', fontsize=10, fontname='Arial')
#plt.savefig("Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/Density_HeLa_Norm_all.svg", format="svg")


# In[12]:


# getting the cases specific to LIPO
HeLa_LIPO = df4[df4["Method"] == 'Lipofectamine']
HeLa_LIPO.head()


# In[13]:


# getting the cases specific to LEPD
HeLa_LEPD = df4[df4["Method"] == 'LEPD']
HeLa_LEPD.head()


# In[30]:


# printing the mean and std of the Normalized intensities 
lipo_Hela=np.asarray(HeLa_LIPO["Norm_all"])
lepd_Hela=np.asarray(HeLa_LEPD["Norm_all"])
print('median_lipo',np.median(lipo_Hela))
print('std_lipo',np.std(lipo_Hela))
print('median_lepd',np.median(lepd_Hela))
print('std_lepd',np.std(lepd_Hela))


# In[17]:


# Levene test to compare variances 
from scipy.stats import levene
stat, p = levene(lipo_Hela,lepd_Hela)
p


# In[5]:


# Load data set with GFP data for k562 cells transfected via both LEPD and LIPO
xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa/Data_1.xlsx')
df5 = pd.read_excel(xls, 'K562')
df5.head(6)


# In[6]:


# Plot distribution of Normalized intensities 
plt.figure(figsize=(4.0,2.5))
color2=['salmon','cornflowerblue']

sns.kdeplot(data=df5, x="Norm_all", hue="Method", multiple="layer", palette=color2, log_scale=None, thresh=0.9, fill=True,bw_adjust = 0.3)


plt.yticks(fontsize = 8)
plt.xticks(fontsize = 8)
plt.xlabel('Intensity(a.u.)', fontsize=10, fontname='Arial')
plt.ylabel('Density', fontsize=10, fontname='Arial')
#plt.savefig("Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/Density_k562_Norm_all.svg", format="svg")


# In[18]:


# getting the cases specific to LIPO
k562_LIPO = df5[df5["Method"] == 'Lipofectamine']
k562_LIPO.head()


# In[19]:


# getting the cases specific to LEPD
k562_LEPD = df5[df5["Method"] == 'LEPD']
k562_LEPD.head()


# In[31]:


# printing the mean and std of the Normalized intensities 
lipo_k562=np.asarray(k562_LIPO["Norm_all"])
lepd_k562=np.asarray(k562_LEPD["Norm_all"])

print('median_lipo',np.median(lipo_k562))
print('std_lipo',np.std(lipo_k562))
print('median_lepd',np.median(lepd_k562))
print('std_lepd',np.std(lepd_k562))


# In[21]:


# Levene test to compare variances 
from scipy.stats import levene
stat, p = levene(lipo_k562,lepd_k562)
p


# In[25]:


xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa/Data_1.xlsx')
df6 = pd.read_excel(xls, 'Sheet1')
df6.head(6)


# In[26]:


df7=df6[(df6['Cell'] == 'HeLa')]
df7.head()


# In[27]:


df8=df6[(df6['Cell'] == 'K562')]


# In[28]:


plt.figure(figsize=(4.0,2.0))
color2=['salmon', 'cornflowerblue']
ax = sns.boxplot(y="Method", x="Norm_all", data=df7, palette=color2,fliersize=0, linewidth=1.0, width=0.8, orient="h")

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

ax = sns.stripplot(y="Method", x="Norm_all", data=df7, palette=color2, dodge=True,linewidth=0.2,edgecolor='black', jitter=True, s=2, orient="h")

# Get the handles and labels. For this example it'll be 2 tuples
# of length 4 each.
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.025, 0.98), loc=2, borderaxespad=0.)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.ylabel('Cell Type', fontsize=18, fontname='Arial')
plt.xlabel('I(a.u.)', fontsize=18, fontname='Arial')
#plt.ylim([0,75000])
#ax.set_yscale('log')
#plt.savefig("Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa_box.svg", format="svg")


# In[29]:


plt.figure(figsize=(4.0,2.0))
color2=['salmon','cornflowerblue']
ax = sns.boxplot(y="Method", x="Norm_all", data=df5, palette=color2,fliersize=0, linewidth=1.0, width=0.8, orient="h")

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

ax = sns.stripplot(y="Method", x="Norm_all", data=df5, palette=color2, dodge=True,linewidth=0.2,edgecolor='black', jitter=True, s=2, orient="h")

# Get the handles and labels. For this example it'll be 2 tuples
# of length 4 each.
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.025, 0.98), loc=2, borderaxespad=0.)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.ylabel('Cell Type', fontsize=18, fontname='Arial')
plt.xlabel('I(a.u.)', fontsize=18, fontname='Arial')
#plt.savefig("Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/K562_box.svg", format="svg")


# In[174]:


plt.figure(figsize=(5.0,5.0))
color2=['salmon', 'cornflowerblue']
ax = sns.boxplot(x="Cell", y="Intensity", hue='Method', data=df6, palette=color2,fliersize=0, linewidth=1.0, width=0.8)

ax = sns.stripplot(x="Cell", y="Intensity", hue='Method', data=df6, palette=color2, dodge=True,linewidth=0.5,edgecolor='black', jitter=True, s=2)

# Get the handles and labels. For this example it'll be 2 tuples
# of length 4 each.
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.025, 0.98), loc=2, borderaxespad=0.)

plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlabel('Cell Type', fontsize=18, fontname='Arial')
plt.ylabel('I(a.u.)', fontsize=18, fontname='Arial')
plt.ylim([0,75000])
#ax.set_yscale('log')
#plt.savefig("Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/Dosage_control.svg", format="svg")


# In[93]:


plt.figure(figsize=(5.0,5.0))
color2=['salmon', 'cornflowerblue']
ax = sns.violinplot(x="Cell", y="Intensity", hue='Method', data=df6, palette=color2,fliersize=0, linewidth=2.5)

#ax = sns.stripplot(x="Cell", y="Intensity", hue='Method', data=df6, palette=color2, dodge=True,linewidth=0.5,edgecolor='black', jitter=True, s=3)

# Get the handles and labels. For this example it'll be 2 tuples
# of length 4 each.
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.025, 0.98), loc=2, borderaxespad=0.)

plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlabel('Cell Type', fontsize=18, fontname='Arial')
plt.ylabel('I(a.u.)', fontsize=18, fontname='Arial')


# In[37]:


xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa/Data_1.xlsx')
df7 = pd.read_excel(xls, 'Norm_HeLa')
df7.head(6)


# In[40]:


lipo_Hela=np.asarray(df7.iloc[:,2])
lepd_Hela=np.asarray(df7.iloc[:,3])
print('mean_lipo',np.mean(lipo_Hela))
print('std_lipo',np.std(lipo_Hela))
print('mean_lepd',np.mean(lepd_Hela))
print('std_lepd',np.std(lepd_Hela))


# In[39]:


from scipy.stats import levene
stat, p = levene(lipo_Hela,lepd_Hela)
p


# In[42]:


xls = pd.ExcelFile('Z:/Papers/LEPD_24_well_plate/Transfection of different cell types/Dossage/HeLa/Data_1.xlsx')
df8 = pd.read_excel(xls, 'Norm_K562')
df8.head(6)


# In[45]:


lipo_k562=np.asarray(df8.iloc[:,2])
lepd_k562=np.asarray(df8.iloc[:,3])

nan_array = np.isnan(lipo_k562)
not_nan_array = ~ nan_array
lipo_k562 = lipo_k562[not_nan_array]


# In[46]:


print('mean_lipo',np.mean(lipo_k562))
print('std_lipo',np.std(lipo_k562))
print('mean_lepd',np.mean(lepd_k562))
print('std_lepd',np.std(lepd_k562))


# In[47]:


stat, p = levene(lipo_k562,lepd_k562)
p


# In[76]:


plt.hist(lepd_k562, density=True, bins=50, color ='red', alpha = 0.5)  # density=False would make counts
plt.hist(lipo_k562, density=True, bins=50, color = "skyblue", alpha = 0.5)
plt.ylabel('Counts')
plt.xlabel('Intensity (a.u.)')


# In[64]:


lepd_gfp=df4.loc[(df4['Plasmid'] == 'GFP') & (df4['Method'] == 'LEPD')]
lipo_gfp=df4.loc[(df4['Plasmid'] == 'GFP') & (df4['Method'] == 'Lipofectamine')]


# In[65]:


lepd_rfp=df4.loc[(df4['Plasmid'] == 'mCherry') & (df4['Method'] == 'LEPD')]
lipo_rfp=df4.loc[(df4['Plasmid'] == 'mCherry') & (df4['Method'] == 'Lipofectamine')]


# In[66]:


gfp1=lepd_gfp.Intensity
gfp1=np.asarray(gfp1)

gfp2=lipo_gfp.Intensity
gfp2=np.asarray(gfp2)

rfp1=lepd_rfp.Intensity
rfp1=np.asarray(rfp1)

rfp2=lipo_rfp.Intensity
rfp2=np.asarray(rfp2)


# In[68]:


plt.hist(np.log10(gfp1), density=True, bins=50, color ='red', alpha = 0.5)  # density=False would make counts
plt.hist(np.log10(gfp2), density=True, bins=50, color = "skyblue", alpha = 0.5)
plt.ylabel('Counts')
plt.xlabel('GFP Intensity')

