#!/usr/bin/env python
# coding: utf-8

# In[9]:


# importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tifffile import imsave
import tifffile as tif


# In[3]:


img1 = cv2.imread('Z:/Research/LEPD_24_well/SNA_B_gal/Confocal/OneDrive_2022-09-27/Processed_images/endocytosis_63x_4-0001.tif')


# In[4]:


alpha = 2.8 # Contrast control (1.0-3.0)
beta = 20 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)

plt.figure(figsize=(7,3.5))
plt.subplot(121), plt.imshow(img1)
plt.subplot(122), plt.imshow(adjusted)


# In[5]:


plt.figure(2, dpi=500)
plt.imshow(adjusted)
#plt.savefig('Z:/Papers/LEPD-Protein\Figures\Figure 2_SNA_protein delivery/SNA2_FITC_1_cropped_B_C_adjusted.png')


# In[15]:


# denoising of image saving it into dst image
dst = cv2.fastNlMeansDenoisingColored(adjusted, None, 25, 10, 21, 7)
  
# Plotting of source and destination image
plt.figure(1, dpi=500)
plt.subplot(121), plt.imshow(adjusted)
plt.subplot(122), plt.imshow(dst)
plt.show()


# In[ ]:


# denoising of image saving it into dst image
dst = cv2.fastNlMeansDenoising(adjusted, None, 25, 10, 21, 7)
  
# Plotting of source and destination image
plt.figure(1, dpi=500)
plt.subplot(121), plt.imshow(adjusted)
plt.subplot(122), plt.imshow(dst)
plt.show()


# In[16]:


plt.figure(3, dpi=500)
plt.imshow(dst)
#plt.savefig('Z:/Research/LEPD_24_well/SNA_B_gal/Confocal/OneDrive_2022-09-27/Processed_images/endocytosis_63x_4-0001_denoised.tif', dpi=500)
tif.imwrite('Z:/Research/LEPD_24_well/SNA_B_gal/Confocal/OneDrive_2022-09-27/Processed_images/endocytosis_63x_4-0001_denoised.tif', dst)


# In[ ]:


alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(dst, alpha=alpha, beta=beta)

plt.figure(2)
plt.subplot(121), plt.imshow(dst)
plt.subplot(122), plt.imshow(adjusted)


# In[ ]:


plt.figure(3)
plt.imshow(adjusted)
#plt.savefig('Z:/Papers/LEPD-Protein\Figures\Figure 2_SNA_protein delivery/SNA2_FITC_1_cropped_adjusted.png')
plt.savefig('Z:/Papers/LEPD-Protein\Figures\FIgure 3_CRISPR\K562/bfp_2_color_ad.tif')


# In[23]:


import numpy as np
from PIL import Image
data = np.random.randint(0, 255, (10,10)).astype(np.uint8)
im = Image.fromarray(dst)
im.save('Z:/Papers/LEPD-Protein\Figures\FIgure 3_CRISPR\K562/bfp_3_cropped_desnoised.tif')


# In[ ]:


pip install pillow


# In[ ]:




