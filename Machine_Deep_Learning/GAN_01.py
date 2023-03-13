#!/usr/bin/env python
# coding: utf-8

# In[76]:


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import normalize
import tifffile as tiff
from patchify import patchify, unpatchify
from skimage.morphology import (erosion, binary_dilation, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table


# In[17]:


train_images = tiff.imread('Q:/BBBC/train_img.tif')


# In[18]:


all_img_patches = []
for img in range(train_images.shape[0]):
    #print(img)     #just stop here to see all file names printed
     
    large_image = train_images[img]
    
    patches_img = patchify(large_image, (256, 256), step=128)  #Step=256 for 256 patches means no overlap, get 12 patches from a single large img
    

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img.astype('float32')) / 255.
            #scaler = MinMaxScaler()
            #single_patch_img= scaler.fit_transform(single_patch_img)
            
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
train_images = normalize(images, axis=1)
train_images = np.expand_dims(train_images, axis=3)


# In[19]:


print(train_images.shape)


# In[33]:


plt.figure(figsize=(6, 6))
k=0
import random
for i in range(3):
    
    for j in range(4): 
        k=k+1
        plt.subplot(3, 4, k)
        test_img_number = random.randint(0, len(train_images))
        test_img = train_images[test_img_number]
        plt.imshow(test_img[:,:,0], cmap='gray')


# In[37]:


from skimage.filters import threshold_otsu


# In[38]:


thresh = threshold_otsu(train_images)
binary = train_images > thresh


# In[39]:


plt.figure(figsize=(6, 6))
k=0
import random
for i in range(3):
    
    for j in range(4): 
        k=k+1
        plt.subplot(3, 4, k)
        test_img_number = random.randint(0, len(binary))
        test_img = binary[test_img_number]
        plt.imshow(test_img[:,:,0], cmap='gray')


# In[110]:


def multi_dil(im, num):
    for i in range(num):
        im = binary_dilation(im)
    return im
def multi_ero(im, num):
    for i in range(num):
        im = erosion(im)
    return im
#multi_dilated = multi_dil(test_img, 2)
#area_closed = area_closing(multi_dilated, 50000)
multi_eroded = multi_ero(binary, 3)
opened = opening(multi_eroded)


# In[ ]:





# In[46]:


import cv2


# In[106]:


test_img_number = random.randint(0, len(binary))
test_img = binary[test_img_number]
#test_img = binary[349]
plt.imshow(test_img[:,:,0], cmap='gray')
print(test_img_number)


# In[107]:


im=test_img


# In[108]:


print(type(multi_eroded))
print(type(multi_eroded))


# In[109]:



def multi_dil(im, num):
    for i in range(num):
        im = binary_dilation(im)
    return im
def multi_ero(im, num):
    for i in range(num):
        im = erosion(im)
    return im
#multi_dilated = multi_dil(test_img, 2)
#area_closed = area_closing(multi_dilated, 50000)
multi_eroded = multi_ero(test_img, 3)
opened = opening(multi_eroded)
plt.subplot(1, 2, 1)
plt.imshow(test_img)
plt.subplot(1, 2, 2)
plt.imshow(opened)


# In[57]:


label_im = label(test_img)
regions = regionprops(label_im)
plt.imshow(label_im)


# In[58]:


np.unique(label_im)


# In[1]:


import tensorflow as tf


# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D, Input

# define simple 3x3 input for this exercise
X = asarray([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

# reshape input data te get it ready for the model (N, x, y, channels)
X = X.reshape((1, X.shape[0], X.shape[1], 1))


"""
Let us define a model. 
#Upsampling size: for size=(2,2) output would be 6,6 since our array size is 3x3 
#and for size=(3,3) output would be 9x9
"""

model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2], 1)))
model.add(UpSampling2D(size = (2,2)))
model.summary()

# Apply the model to our input data
upsampled_X = model.predict(X)

# reshape to just get our x and y
upsampled_X = upsampled_X.reshape((upsampled_X.shape[1],upsampled_X.shape[2]))
print(upsampled_X)

########################################################


# In[16]:


# example of using the transpose convolutional layer
from numpy import asarray
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, Input

# define simple 3x3 input for this exercise
X = asarray([[1, 2, 3, 10],
             [4, 5, 6, 11],
             [7, 8, 9, 12], 
            [13, 14,15,16]])

# reshape input data te get it ready for the model (N, x, y, channels)
X = X.reshape((1, X.shape[0], X.shape[1], 1))
print(np.shape(X))
print(X.reshape(X.shape[1], X.shape[1]))

"""
The Conv2DTranspose upsamples and also performs a convolution. 
Since this includes convolution we need to specify both the size and 
number of filters. We also need to specify stride that gets used for upsampling. 
With a stride of (2,2) the rows and columns of value 0 are inserted to get this stride. 
In this example:
    num_features = 1 (normally we have 64 or 128 or 256 .. etc. )
    kernel size = 1×1
    stride = 2x2 (so a 3x3 image will result as 6x6). 1x1 stride will return the input
Also try stride 3x3 (result will be 9x9 as out input is 3x3 size)
We will initialize kernel by using weights =1. Otherwise it assigns random weights 
and output will not make sense. 
"""

model1 = Sequential()
model1.add(Input(shape=(X.shape[1], X.shape[2], 1)))
model1.add(Conv2DTranspose(1, (2,2), strides=(1,1), kernel_initializer='ones'))
model1.summary()


# Apply the model to our input data
transposed_X = model1.predict(X)
transposed_X = transposed_X.reshape((transposed_X.shape[1],transposed_X.shape[2]))
print(transposed_X)


# In[ ]:




