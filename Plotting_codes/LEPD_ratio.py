#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
from matplotlib import pyplot as plt
import numpy as np
from pylab import rcParams
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['axes.labelpad'] = 3
matplotlib.rcParams['axes.ymargin'] = 0.1
matplotlib.rcParams['axes.xmargin'] = 0

matplotlib.rcParams['axes.edgecolor'] = 'k'

#grids
matplotlib.rcParams['axes.grid'] = False  ## display grid or not
matplotlib.rcParams['axes.grid.axis'] = 'both'    ## which axis the grid should apply to
matplotlib.rcParams['axes.grid.which']= 'both'   ## gridlines at major, minor or both ticks


#ticks
matplotlib.rcParams['xtick.labelsize'] = 6
matplotlib.rcParams['ytick.labelsize'] = 6
matplotlib.rcParams['xtick.major.size'] = 3
matplotlib.rcParams['ytick.major.size'] = 3
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['ytick.major.pad'] = 3
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

#lines
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 2
matplotlib.rcParams['lines.markeredgewidth'] = 0
#matplotlib.rcParams['lines.color'] = 'r'
matplotlib.rcParams["errorbar.capsize"] = 2.5

#### Legend
matplotlib.rcParams['legend.loc'] = 'lower left'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.fontsize'] = 6
matplotlib.rcParams['legend.labelspacing'] = 0.3
matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.handleheight'] = 0.7
matplotlib.rcParams['legend.handletextpad'] = 0.5
matplotlib.rcParams['legend.facecolor'] = 'white'
matplotlib.rcParams['legend.edgecolor'] = 'black'



#saving
matplotlib.rcParams['savefig.pad_inches'] = .5
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['pdf.compression'] = 0

matplotlib.rcParams["figure.figsize"] = [1.3,1.7]

matplotlib.rcParams['savefig.pad_inches'] = .5
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['pdf.compression'] = 0






# In[2]:


xls = pd.ExcelFile('Z:/Research/LCAD Sampling/Nibir/LEPD_24/Ratiometric_9.3.20/Ratio_combined.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
df1.head(6)


# In[3]:


Conc_Ratio=df1.Conc_ratio
Mean=df1.Normalized_mean
SEM=df1.Normalized_SEM
Conc_Ratio = np.asarray(Conc_Ratio)
Mean = np.asarray(Mean)
SEM = np.asarray(SEM)


# In[8]:


#fig, ax = plt.subplots(figsize=(5.0, 5.0))
from matplotlib.pyplot import figure
figure(num=None, figsize=(3.0, 3.0), dpi=80, facecolor='w', edgecolor='k')

plt.errorbar( Conc_Ratio[1:6], Mean[1:6], yerr=SEM[1:6], fmt='o', color='Black', elinewidth=1.5,capthick=1.5, capsize=5)
#ax.errorbar( Conc_Ratio[1:6], Mean[1:6], yerr=SEM[1:6], fmt='o', color='Black', elinewidth=1.5,capthick=1.5, capsize=5)
plt.bar(Conc_Ratio[1:6], Mean[1:6], yerr=SEM[1:6], width= 0.6, align='center', alpha=0.5, color= 'darkgoldenrod', ecolor='black')
#ax.bar(Conc_Ratio[1:6], Mean[1:6], yerr=SEM[1:6], width= 0.6, align='center', alpha=0.5, color= 'darkgoldenrod', ecolor='black')
plt.ylabel('Intensity ratio (GFP : mCherry)',  fontsize=14, fontname='Arial')
#ax.set_ylabel('Intensity ratio (GFP : mCherry)',  fontsize=14, fontname='Arial')
plt.xlabel('Plasmid ratio (GFP : mCherry)',  fontsize=14, fontname='Arial')
#ax.set_xlabel('Plasmid ratio (GFP : mCherry)',  fontsize=14, fontname='Arial')
plt.title('Ratiometric delivery',fontsize=14, fontname='Arial')
#ax.set_title('Ratiometric delivery',fontsize=14, fontname='Arial')
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.ylim(0, 3.2)
#plt.savefig("Z:/Research/LCAD Sampling/Nibir/LEPD_24/Ratiometric_9.3.20/barplot.png", format="png")
plt.savefig("Q:/LEPD_24/Ratiometric_1.svg", format="svg")


# In[ ]:




