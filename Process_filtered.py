#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
from patchify import patchify, unpatchify
import cv2
from skimage import measure, io, img_as_ubyte
from skimage.color import label2rgb, rgb2gray
from tifffile import imsave
import tifffile as tif
import imageio


# In[2]:


# import filtered images 
images = tiff.imread('Z:/Research/LSPR/Data/LSPRi/11.10.22/25nM TNFa/TNFa_1/Arr_filtered.tif')
print(images.shape)
print(type(images))


# In[3]:


# plot the images  
plt.figure(figsize=(14, 14))
n=0
for i in range(0,16):
        n=n+1
        plt.subplot(4,4,n)
        plt.title('Image:%i' %(i+1))
        plt.imshow(images[i,:,:], cmap='gray')
plt.show()


# In[4]:


img_no=5
print("mean =",np.mean(images[img_no,:,:]))
print("std =",np.std(images[img_no,:,:]))
print("min =",np.min(images[img_no,:,:]))
print("max =",np.max(images[img_no,:,:]))


# In[5]:


images_8bit = cv2.normalize(images, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# In[6]:


# Binarizing images 
img_bin=np.empty((images.shape[0],images.shape[1], images.shape[2]))
for i in range(images.shape[0]):
    ret, img_bin[i,:,:] = cv2.threshold(images_8bit[i,:,:],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(figsize=(14, 14)) 
n=0
for i in range(0,16):
    n=n+1
    plt.subplot(4,4,n)
    plt.title('Mask:%i' %(i+1))
    plt.imshow(img_bin[i,:,:], cmap='gray')
plt.show()
print((img_bin.shape))


# In[7]:


index=[]
num_arr = 9
img_siz_fil = np.zeros((images.shape[0],images.shape[1], images.shape[2]))
c = 0
for i in range(images.shape[0]): #images.shape[0]
    
    

# find all of the connected components (white blobs in your image).
# im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    img1 = np.uint8(img_bin[i,:,:])
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img1, connectivity=4) 
# here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
# the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
# you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1
# minimum size of particles we want to keep (number of pixels).
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 500
    max_size = 3000

# output image with only the kept components

    im_result1 = np.zeros((img_bin.shape[1], img_bin.shape[2]))
    
# for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if max_size>sizes[blob]>min_size:
        # see description of im_with_separated_blobs above
            im_result1[im_with_separated_blobs == blob + 1] = 255
            
    img_result2=np.uint8(im_result1)
    nb_blobs1, im_with_separated_blobs1, stats1, _ = cv2.connectedComponentsWithStats(img_result2, connectivity=4)
    if nb_blobs1==num_arr+1:
        index.append(i)
        img_siz_fil[c,:,:] = im_result1
        c=c+1


# In[8]:


count = np.shape(index)
print(img_siz_fil.shape)
print(count)


# In[9]:


print(index)
print(stats1[:, -1])
print(nb_blobs1)


# In[10]:


print(img_bin[0,:,:].dtype)
print(img_siz_fil[0,:,:].dtype)
print(np.min(img_siz_fil))
print(np.max(img_siz_fil))


# In[11]:


count=int(np.asarray(count))


# In[12]:


#Label connected regions of an integer array using measure.label
#Labels each connected entity as one object
#Connectivity = Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
#If None, a full connectivity of input.ndim is used, number of dimensions of the image
#For 2D image it would be 2
bin_label = measure.label(img_siz_fil[:count,:,:], connectivity=images.ndim, background=0)
plt.figure(figsize=(14, 14))
n=0
for i in range(0,16):
        n=n+1
        plt.subplot(4,4,n)
        plt.title('Image:%i' %(i+1))
        plt.imshow(bin_label[i,:,:], cmap='nipy_spectral')
plt.show()


# In[15]:


import pandas as pd
df = pd.DataFrame()
df.head()


# In[16]:


# using region props to extract intesnity values  
for i in range(count):

    props = measure.regionprops_table(bin_label[i,:,:], images[index[i],:,:], 
                          properties=['label','area','mean_intensity'])
    
    df_dictionary = pd.DataFrame([props])
    df = pd.concat([df, df_dictionary],ignore_index=True)
df.tail()


# In[56]:


#df.to_excel("Z:/Research/LSPR/Data/LSPRi/11.10.22/25nM TNFa/TNFa_1/TNFa.xlsx")


# In[17]:


Intensity = df.mean_intensity
print(type(Intensity))


# In[18]:


# transfering intesnity values to an array 
num_steps = count
I_arr=np.empty([num_steps,num_arr])
c=0
for i in range(count):
    c=c+1
    I_arr[i,:]=np.reshape(Intensity[i], (1, num_arr))
print(np.shape(I_arr))


# In[26]:


from matplotlib.pyplot import figure
plt.figure(figsize=(12, 8))

x=np.arange(count)*20
n=0
for i in range(7,8):
#for i in range(0,num_arr):
    n=n+1
    plt.subplot(3,3,n)
    plt.title('Array:%i' %(i+1), fontsize=8)
    plt.scatter(x[3:count], I_arr[3:count,i], c='red', alpha=1.0, marker ='d', s=2.0, linewidths=2.0)
    plt.xlabel('Time (sec)', fontsize=10, fontname='Arial')
    plt.ylabel('Mean Intensity (a.u.)', fontsize=10, fontname='Arial')

    plt.xticks(np.arange(0, x[np.shape(I_arr)[0]-1], step=500), fontsize = 8)
    plt.yticks(fontsize = 10)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.6,
                    hspace=0.6)
    plt.ylim([19700, 20100])
    #plt.savefig('Z:/Research/LSPR/Data/LSPRi/9.21.22/Noggin_25nM_1_1/IvT.png', format="png")
#plt.show()


# In[307]:


import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
x=np.arange(count)*36
ax[0].plot(x[0:count], I_arr[0:count,0])
ax[0].set_title("default axes ranges")

ax[1].plot(x[0:count], I_arr[0:count,1])
ax[1].axis('tight')
ax[1].set_title("tight axes")

ax[2].plot(x[0:count], I_arr[0:count,2])
ax[2].set_ylim([28000, 29000])
ax[2].set_title("custom axes range");


# In[298]:


for e in I_arr:
    if e<1700


# In[ ]:




