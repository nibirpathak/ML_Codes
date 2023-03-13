#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import cv2
import skimage.filters as filters
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from patchify import patchify
from matplotlib import pyplot as plt


# In[20]:



# Load fluorescence image
img = cv2.imread('Q:/siRNA_delivery.tif', cv2.IMREAD_GRAYSCALE)

plt.figure(1)
plt.imshow(img)


# In[23]:


# Noise removal using a filter of choice, try different approaches as results can be image dependent 
blur = cv2.GaussianBlur(img, (25, 25), 0)
plt.figure(2)
plt.imshow(blur)

'''
# Remove noise using median filter
img_filtered = cv2.medianBlur(img_corrected, 3)

plt.figure(3)
plt.imshow(img_filtered)
'''


# In[36]:


# calculate the background using a rolling ball

from cv2_rolling_ball import subtract_background_rolling_ball


radius = 50 # adjust the radius as needed

final_img, background = subtract_background_rolling_ball(blur, radius, light_background=True,
                                     use_paraboloid=False, do_presmooth=True)

# Normalize background

norm_background = cv2.normalize(background, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

#Subtract background
corrected = cv2.absdiff(blur, norm_background)

plt.figure(3)
plt.imshow(corrected)


# In[37]:


# Split image into smaller patches using patchify
patch_size = 256
patches = patchify(blur, (patch_size, patch_size), step=patch_size)


# In[39]:



# Create empty dataframe to store cell information
df_cells = pd.DataFrame(columns=['patch', 'cell_id', 'cell_intensity', 'cell_area', 'cell_perimeter'])

# Loop through patches and perform cell segmentation and analysis
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        # Perform segmentation using watershed algorithm
        patch = patches[i,j,:,:]
        patch_smoothed = cv2.GaussianBlur(patch, (3,3), 0)
        markers = filters.threshold_minimum(patch_smoothed) * np.ones_like(patch_smoothed)
        markers[peak_local_max(patch_smoothed, indices=False, exclude_border=False)] = 0
        labels = watershed(filters.gaussian(patch_smoothed, sigma=1), markers, mask=patch_smoothed)
        
        # Extract cell features using regionprops
        for region in regionprops(labels):
            cell_id = region.label
            cell_intensity = region.mean_intensity
            cell_area = region.area
            cell_perimeter = region.perimeter
            df_cells = df_cells.append({'patch': (i,j), 'cell_id': cell_id, 'cell_intensity': cell_intensity, 'cell_area': cell_area, 'cell_perimeter': cell_perimeter}, ignore_index=True)


# In[ ]:


# Normalize and reduce cell features using PCA
scaler = StandardScaler()
X = scaler.fit_transform(df_cells[['cell_intensity', 'cell_area', 'cell_perimeter']])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_cells['cell_pca1'] = X_pca[:,0]
df_cells['cell_pca2'] = X_pca[:,1]

# Display results
print(df_cells.head())

