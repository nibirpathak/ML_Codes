#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[3]:


xls = pd.ExcelFile('D:/Data set with proper imperfection/C1_C2_C3_Geom_tilts_1_2.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
df1.head(10)


# In[4]:


# Check and set sample sizes 
samples=np.size(df1.a1)
print(samples)
train_size=int(0.8*samples)
test_size=samples-train_size
print(train_size, test_size)
valid_size=int(1*samples)-int(1*train_size)
print(valid_size)


# In[5]:


class TrainDataset():
    def __init__(self): 
        
    
        xls = pd.ExcelFile('D:/Data set with proper imperfection/C1_C2_C3_Geom_tilts_1_2.xlsx')
        df1 = pd.read_excel(xls, 'Sheet1')
        df1 = df1.sample(frac=1)       # shuffle rows 
        
        x=df1.iloc[0:int(1*samples),0:4]     # all 10 params for all data
        y=df1.iloc[0:int(1*samples),4:10]    # twists and tilts for all data np.r_[4,6:8]        
        
        x=np.asarray(x)
        y=np.asarray(y)

        #x_sc=scaler.fit_transform(x)       # minmax scaling data
        #y_sc=scaler.fit_transform(y)
   
        #x=np.asarray(x_sc[0:samples]) # params for training data 
        #y=np.asarray(y_sc[0:samples]) # twists and tilts for training data 
        
        X=np.asarray(x) # params for data 
        Y=np.asarray(y) # labels for data
        
        Y=Y.reshape((int(1*samples),6))  # 'samples' intead of value
        
        self.X=torch.tensor(X, dtype=torch.float32) # converting training data to tensor 
        self.Y=torch.tensor(Y, dtype=torch.float32)
        
        self.X=self.X.view(int(1*samples),4)    # adjusting the shape of the traning data 
        self.Y=self.Y.view(int(1*samples),6)
        
        self.X=self.X.to(device)  #sending tesnors to GPU
        self.Y=self.Y.to(device)
        print(self.X)
        print(self.Y)
        
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_splits(self):
        train_size = round(0.8*len(self.X))
        test_size = len(self.X)-train_size
        print(train_size)
        print(test_size)
        return random_split(self, [train_size, test_size])


# In[6]:


tmp = TrainDataset()
print(tmp.__len__())
print(tmp.get_splits())


# In[7]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer
        hidden_1 = 512
        hidden_2 = 512
        #hidden_3 = 600
        #hidden_4 = 200
        #hidden_5 = 32
        # linear layer (#features -> hidden_1)
        
        self.fc1 = nn.Linear(4, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        #self.fc3 = nn.Linear(hidden_2, hidden_3)
        #self.fc4 = nn.Linear(hidden_3, hidden_4)
        #self.fc5 = nn.Linear(hidden_4, hidden_5)
        
        # output layer (n_hidden -> #labels)
        self.fc6 = nn.Linear(hidden_2, 6)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # add output layer
        x = self.fc6(x)
        return x


# In[8]:


# prepare the dataset
def prepare_data():
    # load the dataset
    dataset = TrainDataset()
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=20, shuffle=True)
    test_dl =  DataLoader(train, batch_size=train_size, shuffle=True)
    valid_dl = DataLoader(test, batch_size=test_size, shuffle=False)
    return train_dl, test_dl, valid_dl


# In[9]:


def train_model(train_dl, net):

    epochs=500
    epoch_loss=[]
    
    # training 
    for e in range(epochs):
        running_loss=0
        k=0
        # batch training
        for features, labels in train_dl:
        
            features, labels=features.type(torch.FloatTensor), labels.type(torch.FloatTensor)
        
            features, labels = features.to(device), labels.to(device)  #sending tesnors to GPU

            output = net(features)     # input x and predict based on x

            loss = loss_func(output, labels)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        #  update weights

            running_loss=running_loss+loss.item()
            #k=k+1
       
        epoch_loss.append(running_loss/len(train_dl))
        #print(k)
    epoch = list(range(1, epochs+1))
    plt.figure(figsize=(2.0,1.5))
    plt.figure(1)
    plt.plot(epoch,epoch_loss,'o',color='brown', markersize=1.0)
    plt.xlabel("Number of epochs",fontsize=10)
    plt.ylabel("Training loss", fontsize=10)
    plt.xticks(fontsize = 10,fontname='Arial')
    plt.yticks(fontsize = 10, fontname='Arial')
    


# In[10]:


# training and validation accuracy 
def test_model(test_dl, net):
    
    for features, labels in test_dl:
        features, labels=features.type(torch.FloatTensor), labels.type(torch.FloatTensor)
        features, labels = features.to(device), labels.to(device)
        output = net(features)
        break
    labels=labels.cpu()  #sending tesnors to CPU for converting to numpy 
    output=output.cpu()
    y_a=labels.numpy()
    y_p=output.data.numpy()
    from sklearn.metrics import r2_score
    print("r2 score = ", r2_score(y_a[:,0], y_p[:,0]))
    print("r2 score = ", r2_score(y_a[:,1], y_p[:,1]))
    print("r2 score = ", r2_score(y_a[:,2], y_p[:,2]))
    print("r2 score = ", r2_score(y_a[:,3], y_p[:,3]))
    print("r2 score = ", r2_score(y_a[:,4], y_p[:,4]))
    print("r2 score = ", r2_score(y_a[:,5], y_p[:,5])) 

    plt.figure(figsize=(2.0,2.0))
    plt.figure(1)
    plt.plot(y_a[:,0], y_p[:,0],'o', color='darkkhaki', markersize=1.0)
    plt.xlabel('True ' r'$\theta_1$', fontsize=10, fontname='Arial')
    plt.ylabel('Predicted ' r'$\theta_1$', fontsize=10, fontname='Arial')
    plt.xticks(np.arange(-25, 20, 10), fontsize = 8,fontname='Arial')
    plt.yticks(np.arange(-25, 20, 10), fontsize = 8, fontname='Arial')
    plt.show()
    
    plt.figure(figsize=(2.0,2.0))
    plt.figure(1)
    plt.plot(y_a[:,1], y_p[:,1],'o', color='darkkhaki', markersize=1.0)
    plt.xlabel('True ' r'$\theta_2$', fontsize=10, fontname='Arial')
    plt.ylabel('Predicted ' r'$\theta_2$', fontsize=10, fontname='Arial')
    plt.xticks(np.arange(-25, 20, 10), fontsize = 8,fontname='Arial')
    plt.yticks(np.arange(-25, 20, 10), fontsize = 8, fontname='Arial')
    plt.show()
    


# In[11]:


train_dl, test_dl, valid_dl = prepare_data()
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
#sending model to GPU
net.to(device)


# In[12]:


# train the model
net=net.train()
train_model(train_dl, net)


# In[13]:


# evaluate the model
#net=net.eval()
test_model(test_dl, net)


# In[14]:


# evaluate the model
test_model(valid_dl, net)


# In[15]:


class Inv_TrainDataset():
    def __init__(self): 
        
    
        xls = pd.ExcelFile('D:/Data set with proper imperfection/C1_C2_C3_Geom_tilts_1_2.xlsx')
        df1 = pd.read_excel(xls, 'Sheet1')
        df1 = df1.sample(frac=1)       # shuffle rows 
        
        x=df1.iloc[0:int(1*samples),4:10]     # twists and tilts for all data np.r_[4,6:8]
        y=df1.iloc[0:int(1*samples),0:4]    #   Params      

        x=np.asarray(x)
        y=np.asarray(y)

        #x_sc=scaler.fit_transform(x)       # minmax scaling the test data
        #y_sc=scaler.fit_transform(y)
   
        #x=np.asarray(x_sc[0:samples]) # params for training data 
        #y=np.asarray(y_sc[0:samples]) # twists and tilts for training data 
        
        X=np.asarray(x) # params for data 
        Y=np.asarray(y) # labels for data
        
        Y=Y.reshape((int(1*samples),4))  # 'samples' intead of value
        
        self.X=torch.tensor(X, dtype=torch.float32) # converting training data to tensor 
        self.Y=torch.tensor(Y, dtype=torch.float32)
        
        self.X=self.X.view(int(1*samples),6)    # adjusting the shape of the traning data 
        self.Y=self.Y.view(int(1*samples),4)
        
        self.X=self.X.to(device)  #sending tesnors to GPU
        self.Y=self.Y.to(device)
        print(self.X)
        print(self.Y)
        
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_splits(self):
        train_size = round(0.8*len(self.X))
        test_size = len(self.X)- train_size
        print(train_size)
        print(test_size)
        return random_split(self, [train_size, test_size])


# In[16]:


tmp = Inv_TrainDataset()
print(tmp.__len__())


# In[17]:


class Inv_Net(nn.Module):
    def __init__(self):
        super(Inv_Net, self).__init__()
        # number of hidden nodes in each layer
        hidden_1 = 128
        hidden_2 = 256
        #hidden_3 = 600
        #hidden_4 = 200
        #hidden_5 = 32
        # linear layer (#features -> hidden_1)
        
        self.fc1 = nn.Linear(6, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        #self.fc3 = nn.Linear(hidden_2, hidden_3)
        #self.fc4 = nn.Linear(hidden_3, hidden_4)
        #self.fc5 = nn.Linear(hidden_4, hidden_5)
        
        # output layer (n_hidden -> #labels)
        self.fc6 = nn.Linear(hidden_2, 4)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # add output layer
        x = self.fc6(x)
        return x


# In[18]:


# prepare the inv_dataset
def Inv_prepare_data():
    # load the dataset
    dataset = Inv_TrainDataset()
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    Inv_train_dl = DataLoader(train, batch_size=3000, shuffle=True)
    Inv_test_dl =  DataLoader(train, batch_size=train_size, shuffle=False)
    Inv_valid_dl = DataLoader(test, batch_size=test_size, shuffle=False)
    return Inv_train_dl, Inv_test_dl, Inv_valid_dl


# In[19]:


def Inv_train_model(Inv_train_dl, Inv_net, net):

    epochs=1500
    epoch_loss=[]
    
    # training 
    for e in range(epochs):
        running_loss=0
        k=0
        # batch training
        for features, labels in Inv_train_dl:
        
            features, labels=features.type(torch.FloatTensor), labels.type(torch.FloatTensor)
        
            features, labels = features.to(device), labels.to(device)  #sending tesnors to GPU

            output = Inv_net(features)     # input x and predict based on x

            #loss = loss_func(output, labels)     # must be (1. nn output, 2. target)
            if e<250:
                lamda = 8.0
                mu = 0.05
            else: 
                lamda = 8.00
                mu = 0.05                
            loss = mu*loss_func(net(output), features)+lamda*loss_func(output, labels)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        #  update weights

            running_loss=running_loss+loss.item()
            #k=k+1
       
        epoch_loss.append(running_loss/len(Inv_train_dl))
        #print(k)
    epoch = list(range(1, epochs+1))
    plt.figure(figsize=(2.0,1.5))
    plt.figure(1)
    plt.plot(epoch,epoch_loss,'o',color='saddlebrown', markersize=1.0)
    plt.xlabel("Number of epochs",fontsize=10)
    plt.ylabel("Training loss", fontsize=10)
    plt.xticks(fontsize = 10,fontname='Arial')
    plt.yticks(fontsize = 10, fontname='Arial')


# In[20]:


# training and validation accuracy 
def Inv_test_model(Inv_test_dl, Inv_net, net):
    
    for features, labels in Inv_test_dl:
        features, labels=features.type(torch.FloatTensor), labels.type(torch.FloatTensor)
        features, labels = features.to(device), labels.to(device) #labels are actual params, features are actual angles 
        output = Inv_net(features)  # angles convered to predicted geom params  
        
        
        output = output.type(torch.FloatTensor)
        
        output = output.to(device)   #sending tesnors to GPU
        
        re_output = net(output) # geom params to predicted angles 
        
        
        break
    labels=labels.cpu()  #sending tesnors to CPU for converting to numpy , actual params
    features=features.cpu()  # actual angles 
    output=output.cpu()      # predicted params
    re_output=re_output.cpu() # reconstructed angles by f-nn
    
    y_a=labels.numpy()
    y_p=output.data.numpy()
    angles_a=features.data.numpy()
    angles_p=re_output.data.numpy()
    
    
    from sklearn.metrics import r2_score
    print("r2 score = ", r2_score(y_a[:,0], y_p[:,0]))
    print("r2 score = ", r2_score(y_a[:,1], y_p[:,1]))
    print("r2 score = ", r2_score(y_a[:,2], y_p[:,2]))
    print("r2 score = ", r2_score(y_a[:,3], y_p[:,3]))
    
    
    print("r2 score = ", r2_score(angles_a[:,0], angles_p[:,0]))
    print("r2 score = ", r2_score(angles_a[:,1], angles_p[:,1]))
    print("r2 score = ", r2_score(angles_a[:,2], angles_p[:,2]))
    print("r2 score = ", r2_score(angles_a[:,3], angles_p[:,3]))
    print("r2 score = ", r2_score(angles_a[:,4], angles_p[:,4]))
    print("r2 score = ", r2_score(angles_a[:,5], angles_p[:,5]))
    
    

    plt.figure(figsize=(1.2,1.2))
    plt.figure(1)
    plt.plot(y_a[:,0], y_p[:,0],'o', color='darkolivegreen', markersize=1.0)
    plt.xlabel('True $a_1$', fontsize=10, fontname='Arial')
    plt.ylabel('Predicted $a_1$', fontsize=10, fontname='Arial')
    #plt.xticks(np.arange(0, 3.0, 0.5), fontsize = 8,fontname='Arial')
    #plt.yticks(np.arange(0, 3.0, 0.5), fontsize = 8, fontname='Arial')
    plt.xticks(fontsize = 8,fontname='Arial')
    plt.yticks(fontsize = 8, fontname='Arial')
    plt.show()

    
    plt.figure(figsize=(1.2,1.2))
    plt.figure(1)
    plt.plot(angles_a[:,5], angles_p[:,5],'o', color='chocolate', markersize=1.0)
    plt.xlabel('Desired ' r'$\psi_2^{2}$', fontsize=10, fontname='Arial')
    plt.ylabel('Reconstructed ' r'$\psi_2^{2}$', fontsize=10, fontname='Arial')
    plt.xticks(fontsize = 8,fontname='Arial')
    plt.yticks(fontsize = 8, fontname='Arial')   
    plt.show()    
    
    


# In[21]:


Inv_train_dl, Inv_test_dl, Inv_valid_dl = Inv_prepare_data()
Inv_net = Inv_Net()
optimizer = torch.optim.Adam(Inv_net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
#sending model to GPU
Inv_net.to(device)


# In[22]:


Inv_net=Inv_net.train()
#net=net.eval()
Inv_train_model(Inv_train_dl, Inv_net, net)


# In[23]:


# evaluate the model
#Inv_net=Inv_net.eval()
Inv_test_model(Inv_test_dl, Inv_net, net)


# In[24]:


# evaluate the model
Inv_test_model(Inv_valid_dl, Inv_net, net)


# In[ ]:




