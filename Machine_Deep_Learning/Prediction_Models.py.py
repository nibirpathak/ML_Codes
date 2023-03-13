#!/usr/bin/env python
# coding: utf-8

# In[113]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.spatial import distance
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model
import xlsxwriter 


# In[114]:


xls = pd.ExcelFile('Q:/MLRefined/Project/divorce/Divorce_data.xlsx')
df = pd.read_excel(xls, 'bos')
df.head()


# In[115]:


n, bins, patches = plt.hist(x=df.Atr3, bins=[-0.5,0.5,1.5,2.5,3.5,4.5], color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Atr1')


# In[116]:


X = df.iloc[:,0:54]  #independent columns
y = df.iloc[:,54]


# In[117]:


X_a=np.asarray(X)
y_a=np.asarray(y)


# In[118]:


# Chi-Squared test to see which are more independent from the output, higher the score greater is the dependence

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfpvalues= pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
featureScores.columns = ['Specs','Score','p-value']


# In[119]:


Feature_best=featureScores.sort_values(by=['Score'],ascending=False)
Feature_best.head(10)


# In[92]:


# X_reduced3 are the features selectd after Chi-sqaured test

X_reduced3=np.zeros((len(y),6))
X_reduced3[:,0]=X_a[:,35]
X_reduced3[:,1]=X_a[:,39]
X_reduced3[:,2]=X_a[:,34]
X_reduced3[:,3]=X_a[:,18]
X_reduced3[:,4]=X_a[:,8]
X_reduced3[:,5]=X_a[:,4]


# In[104]:


#  Sequential Forward selection: selection is based on which features give best model accuracy when included 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=58,class_weight=None)
LR = LogisticRegression(max_iter=1000000,class_weight=None)
svm = SVC(kernel='rbf',C=1,class_weight=None)
sfs = SFS(rf, 
          k_features=6, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=10)
sfs = sfs.fit(X, y)
sfs.k_feature_idx_
# takes longer to run as it as to run the model multiple times 


# In[97]:


# setting the reduced features after Sequential Forward selection 
i=0
x_reduced1=np.zeros((len(y),6))
for item in sfs.k_feature_idx_: 
    x_reduced1[:,i]=X_a[:,item]
    i=i+1
# selecting features common to all the models 
X_SFS=X_reduced3=np.zeros((len(y),3))
X_SFS[:,0]=X_a[:,1]
X_SFS[:,1]=X_a[:,6]
X_SFS[:,2]=X_a[:,17]


# In[122]:


# recursive feature elimination with cross validation
from sklearn.feature_selection import RFECV

rf=RandomForestClassifier(n_estimators=58,class_weight=None)
LR = LogisticRegression(max_iter=1000000,class_weight=None)
svm = SVC(kernel='rbf',C=1,class_weight=None)

estimator = rf
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y) 

k=0
idx=[]
for i, val in enumerate(selector.ranking_):
        if val == 1:
            idx.append(i)
            
print(idx)


# In[106]:


i=0
X_RFECV=np.zeros((len(y),3))
for item in idx: 
    X_RFECV[:,i]=X_a[:,item]
    i=i+1


# In[82]:


# Selection is based on eliminating features with low weights gotten after training the model

from sklearn.feature_selection import RFE
from sklearn.svm import SVC
#estimator = SVC(kernel='linear', C=0.01,class_weight=None, gamma='auto',decision_function_shape='ovo')
#estimator =  LogisticRegression(max_iter=1000000,class_weight=None)
estimator =  RandomForestClassifier(n_estimators=58,class_weight=None)
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X, y)

k=0
idx=[]
for i, val in enumerate(selector.ranking_):
        if val == 1:
            idx.append(i)
            
print(idx)


# In[64]:


# setting the reduced features after Recursive feature elimination

i=0
x_reduced2=np.zeros((len(y),6))
for item in idx: 
    x_reduced2[:,i]=X_a[:,item]
    i=i+1


# In[65]:


'''
from scipy.stats import spearmanr
for i in X.columns:
    print(spearmanr(X[i],y))
'''


# In[95]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000000,C=1.0,class_weight=None)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=58,class_weight=None)
from sklearn.neural_network import MLPClassifier
layer_size={100,50,20}
rf


# In[120]:


from sklearn.model_selection import cross_val_score 
SVM_score=cross_val_score(SVC(kernel='rbf',C=1,class_weight=None), X_reduced3, y, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, X_reduced3, y, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, X_reduced3, y, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),X_reduced3, y, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# In[121]:


# Model accuracy using 10-fold validation on the data having the whole feature set

from sklearn.model_selection import cross_val_score 
SVM_score=cross_val_score(SVC(kernel='rbf',C=1,class_weight=None), X, y, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, X, y, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, X, y, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),X, y, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# In[ ]:




