#!/usr/bin/env python
# coding: utf-8

# In[22]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.spatial import distance
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model
import scipy.interpolate as interpolate
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import make_lsq_spline, BSpline
import xlsxwriter 
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


# In[23]:


xls = pd.ExcelFile('Q:/Kirigami/Solar Panel/Center/PCA_Kmeans/Model_input.xlsx')
df = pd.read_excel(xls, 'Sheet1')


# In[24]:


df.head()


# In[25]:


Features=df.drop('Label',axis='columns')
Features_scaled=Features


# In[26]:


from sklearn.preprocessing import MinMaxScaler
Mscaler = MinMaxScaler()
df1=Mscaler.fit_transform(Features)
for i in range(10):
    a=df1[:,i]
    b=np.reshape(a,(950,1))
    Features_scaled.iloc[:,[i]]=b
Features_scaled.head()


# In[27]:



X = Features_scaled.iloc[:,0:10]  #independent columns
y = df.iloc[:,10] 

'''
xdata=np.asarray(Features_scaled.iloc[:,:])
ydata=np.asarray(df.Label)
'''
xdata = Features_scaled[['a2','a1','a3','a4']]  #independent columns
ydata = df.iloc[:,10] 
xdata =np.asarray(xdata)
ydata =np.asarray(ydata)


X_a=np.asarray(X)
y_a=np.asarray(y)


# In[28]:


dataset=Features_scaled
dataset = dataset.assign(Label = df.iloc[:,10]) 
dataset.head()


# In[46]:


# Recursive feature elimination: but assumes a linear relation
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
#estimator = SVC(kernel='linear', C=0.01,class_weight=None, gamma='auto',decision_function_shape='ovo')
estimator = RandomForestClassifier(n_estimators=150,class_weight=None)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
selector.ranking_


# In[30]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X_a,y_a)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']


# In[31]:


Chi_sq=featureScores.sort_values(by=['Score'],ascending=False)
featureScores


# In[33]:


print(featureScores.nlargest(6,'Score'))


# In[34]:


# Selection is based on eliminating features with low weights gotten after training the model
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
estimator = SVC(kernel='linear', C=0.01,class_weight=None, gamma='auto',decision_function_shape='ovo')
#estimator =  LogisticRegression(max_iter=1000000,class_weight=None)
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X, y)

idx=[]
for i, val in enumerate(selector.ranking_):
        if val == 1:
            idx.append(i)
            
print(idx)


# In[35]:


# selection is based on which features give best model accuracy when included 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
#LR = LogisticRegression(max_iter=1000000,class_weight=None)
svm = SVC(kernel='linear', C=0.01,class_weight=None, gamma='auto',decision_function_shape='ovo')
sfs = SFS(svm, 
          k_features=6, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=10)
sfs = sfs.fit(X, y)
sfs.k_feature_idx_


# In[18]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier( n_estimators=100)
model.fit(X,y)


# In[19]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[53]:


from scipy.stats import spearmanr
for i in X.columns:
    print(spearmanr(X[i],y))
    
    


# In[390]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = dataset.corr(method='spearman')
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train,y_train)
Knn_scores = cross_val_score(knn, X_a, y_a, cv=5)
print('accuracy of KNN = %s' %(np.mean(Knn_scores)))


# In[45]:


layer_size={100,15,10}
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', activation= 'relu',alpha=0.0001,hidden_layer_sizes=layer_size, random_state=1, 
max_iter=20000, early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2)
clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
#Out=clf.predict(xdata)


# In[44]:


print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))




# In[263]:


'''
# using confusion matrix to check accuracy
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_pred, y_test)

def accuracy(ConMatrix):
    T=ConMatrix.trace()
    ES= ConMatrix.sum()
    return float(T/ES)
print("Accuracy of MLPClassifier : ", accuracy(cm))
'''


# In[39]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=160,class_weight=None)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[40]:


from sklearn.svm import SVC
svm=SVC(kernel='rbf',degree=3, C=100,class_weight=None, gamma='auto',decision_function_shape='ovo')
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[41]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000000,solver='sag',class_weight=None)


# In[42]:


# K-fold cross validation
from sklearn.model_selection import cross_val_score 
SVM_score=cross_val_score(SVC(kernel='rbf',C=100,gamma='auto',class_weight=None), xdata, ydata, cv=10)
print('accuracy of SVM = %s' %(np.mean(SVM_score)))
LR_score=cross_val_score(LR, xdata, ydata, cv=10)
print('accuracy of LR = %s' %(np.mean(LR_score)))
RF_score=cross_val_score(rf, xdata, ydata, cv=10)
print('accuracy of RF = %s' %(np.mean(RF_score)))
NN_score=cross_val_score(MLPClassifier(solver='lbfgs', activation= 'relu',
         alpha=0.01,hidden_layer_sizes=layer_size, random_state=1,max_iter=20000, 
          early_stopping=False, n_iter_no_change=100, tol=0.001, validation_fraction=0.2),xdata, ydata, cv=5)
print('accuracy of NN = %s' %(np.mean(NN_score)))


# In[24]:


# Model parameters for different models for Gridsearch
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [0.001,0.01,1,10,100,500],
            'kernel': ['rbf','linear','sigmoid']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [220,300,160,200]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [0.001,0.01,1,10,50], 'solver':['sag', 'saga','lbfgs'], 'max_iter':[10000]
        }
    },
    'KNN' : {
        'model': KNeighborsClassifier(n_neighbors = 15),
        'params': {
            'n_neighbors': [5,10,20,30], 'weights':['uniform', 'distance']
        }
    }
}


# In[26]:


scores = []
from sklearn.model_selection import GridSearchCV
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(xdata,ydata)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:


from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(MLPClassifier(), {'max_iter':[50000],'solver':['lbfgs'],'alpha':[0.0001,0.001,0.01,1,10,100],
    'hidden_layer_sizes':[o for o in product((20,15,5,),repeat=3)],
    'activation': ['relu']}, cv=5, return_train_score=False)
clf.fit(X_a, y_a)


# In[ ]:



df = pd.DataFrame(clf.cv_results_)
df[['params','mean_test_score']]


# In[379]:


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[412]:


from sklearn.model_selection import StratifiedKFold
folds=StratifiedKFold(n_splits=10)


# In[429]:



layer_size={20,10,5}
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', activation= 'relu',alpha=0.1,hidden_layer_sizes=layer_size, random_state=None, 
max_iter=200000, early_stopping= False, n_iter_no_change=1000, tol=0.0001)

from sklearn import svm, datasets

scores_NN=[]
scores_SVM=[]
scores_LR=[]
scores_RF=[]
for train_index, test_index in folds.split(X_a,y_a):
    X_train, X_test = X_a[train_index], X_a[test_index]
    y_train, y_test = y_a[train_index], y_a[test_index]
    
    sLR=get_score(LogisticRegression(C=1,max_iter=1000000, solver='lbfgs'),X_train, X_test, y_train, y_test)
    scores_LR.append(sLR)
    
    rfs=get_score(RandomForestClassifier(n_estimators=160,class_weight=None),X_train, X_test, y_train, y_test)
    scores_RF.append(rfs)
    
    sSVM=get_score(SVC(kernel='rbf',C=100,gamma='auto',class_weight=None),X_train, X_test, y_train, y_test)
    scores_SVM.append(sSVM)
    
    #sNN=get_score(clf,X_train, X_test, y_train, y_test)
    #scores_NN.append(sNN)


# In[430]:


Mean_SVM=np.mean(scores_SVM)*100
Mean_LR=np.mean(scores_LR)*100
Mean_RF=np.mean(scores_RF)*100
print('Accuracy of SVM = %s%%'%(Mean_SVM))
print('Accuracy of Logistic Regression =%s%%'%(Mean_LR))
print('Accuracy of RF =%s%%'%(Mean_RF))


# In[166]:


print(scores_SVM)
print(scores_LR)
#print(scores_NN)


# In[246]:


p=np.arange(10,15)
np.size(p)


# In[ ]:





# In[ ]:




