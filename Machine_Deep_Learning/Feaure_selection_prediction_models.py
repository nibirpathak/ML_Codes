#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.spatial import distance
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model
import xlsxwriter 

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 


# In[6]:


# load data set
xls = pd.ExcelFile('Q:/MLRefined/Project/divorce/Divorce_data.xlsx')
df = pd.read_excel(xls, 'bos')
df.head()


# In[7]:


X = df.iloc[:,0:54]  #independent columns
y = df.iloc[:,54]

X_a=np.asarray(X)
y_a=np.asarray(y)


# In[8]:


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


# In[9]:


Feature_best=featureScores.sort_values(by=['Score'],ascending=False)
Feature_best.head(10)


# In[10]:


# X_reduced3 are the features selectd after Chi-sqaured test

X_reduced3=np.zeros((len(y),6))
X_reduced3[:,0]=X_a[:,35]
X_reduced3[:,1]=X_a[:,39]
X_reduced3[:,2]=X_a[:,34]
X_reduced3[:,3]=X_a[:,18]
X_reduced3[:,4]=X_a[:,8]
X_reduced3[:,5]=X_a[:,4]


# In[ ]:


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


# In[ ]:


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


# In[11]:


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


# In[23]:


# load in dataset, reloading the data as it was written by another group member 
with open("divorce.csv","r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ",")
    lines = []
    for row in csvreader:
        lines.append(row[0].split(';'))
data = np.asarray(lines[1:172],dtype=np.float32)
x = data[:,:-1]
y = data[:,-1]


# In[24]:


# Lasso
nData = len(x)
alphaGrid = np.logspace(-3,1,30)
errGrid = np.zeros((nData,len(alphaGrid)))
nCovars = np.zeros_like(alphaGrid)
lassoObj = Lasso(alpha = 0.1, max_iter = 3000)

for ii, alpha in enumerate(alphaGrid):
#     print(f"\n Training with alph = {alpha}!")
    
    ## Set new alpha
    lassoObj.set_params(alpha = alpha)
    
    # Get training and test data: Leave one out cross validation        
    for jj in range(nData):
        trainX = x.copy()
        trainX = np.delete(trainX,jj,0)
        trainY = np.delete(y,jj,0)
        testX = x[jj]
        testY = y[jj]
        
        lassoObj.fit(trainX,trainY)
        yPred = np.dot(testX, lassoObj.coef_)
        
        # Save error
        errGrid[jj,ii] = np.sum((testY-yPred)**2)
    
    # Save number of nonzero covariates
    nCovars[ii] = np.sum(lassoObj.coef_ != 0)

meanErr = np.mean(errGrid,axis = 0)
bestAlpha = alphaGrid[np.argmin(meanErr)]

plt.plot(np.log10(alphaGrid),errGrid.T, 'ob',alpha = 0.2)
plt.plot(np.log10(alphaGrid),meanErr,'-r',label = 'mean')
plt.yscale('log')
#plt.vlines(np.log10(bestAlpha),10**-7,10,'g',label ='Min Error')
plt.legend()
plt.yscale('log')
plt.xlabel('lambdas')
plt.ylabel('MSE')

plt.figure()
plt.plot(np.log10(alphaGrid),nCovars)
plt.xlabel('lambdas')
plt.ylabel('Number of nonzero coefficients')
#plt.vlines(np.log10(bestAlpha),5,40,'g',label ='Min Error')

bestLasso = Lasso(alpha = bestAlpha, max_iter = 3000)
bestLasso.fit(x,y)
print(bestAlpha,np.sum(bestLasso.coef_ != 0))


# In[25]:


bestLasso = Lasso(alpha = 0.09, max_iter = 3000)
bestLasso.fit(x,y)
np.sum(bestLasso.coef_ != 0)
bestLasso.coef_


# In[26]:


features = np.arange(54)+1
dfcovar = pd.DataFrame(bestLasso.coef_)
dfcolumns = pd.DataFrame(features)

featureCovar = pd.concat([dfcolumns,dfcovar],axis=1)
featureCovar.columns = ['Feature','Covariate']  #naming the dataframe columns
print(featureCovar.nlargest(6,'Covariate'))  #print 10 best features


# In[27]:


bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Chi2 Score']  #naming the dataframe columns
print(featureScores.nlargest(6,'Chi2 Score'))  #print 10 best features


# ## Select data features chosen by different methods

# In[28]:


x_Lasso = x[:,bestLasso.coef_>0]
feat = np.asarray([35,29,34,18,8,4])
x_chi2 = x[:,feat]


# In[29]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000000,class_weight=None)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=58,class_weight=None)
from sklearn.neural_network import MLPClassifier
layer_size={100,50,20}


# ### Model accuracy using 10-fold validation on the data having the whole feature set

# In[30]:


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


# ### Model accuracy using features from RFECV

# In[31]:


i=0
X_RFECV=np.zeros((len(y),3))
for item in idx: 
    X_RFECV[:,i]=X_a[:,item]
    i=i+1

SVM_score=cross_val_score(SVC(kernel='rbf',C=1,class_weight=None), X_RFECV, y, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, X_RFECV, y, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, X_RFECV, y, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),X_RFECV, y, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# ### Model accuracy using features from Chi-2 method

# In[32]:


SVM_score=cross_val_score(SVC(kernel='rbf',C=1,class_weight=None), x_chi2, y, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, x_chi2, y, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, x_chi2, y, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),x_chi2, y, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# ### Model accuracy using features from LASSO

# In[33]:


SVM_score=cross_val_score(SVC(kernel='rbf',C=1,class_weight=None), x_Lasso, y, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, x_Lasso, y, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, x_Lasso, y, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),x_Lasso, y, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# In[ ]:





# In[ ]:




