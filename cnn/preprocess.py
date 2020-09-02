# %% normalize all pixels from [0,255] to [0,1]
from numpy import asarray
from PIL import Image
import os
import numpy as np


# %%
from keras.preprocessing.image import array_to_img, img_to_array, load_img
fp = '../COVID-img-data/NORMAL'

arr = []
q = 0
for fname in os.listdir(fp):
    image = load_img(fp+'/'+fname)
    pixels = img_to_array(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    arr.append(pixels)
    q += 1
    if q>30:
        break
# %%
for i in range(30):
    print(arr[i].shape)
    
# %%
# remove array elements without 3 color channels
# new_arr = [elem for elem in arr if elem.shape == (1024, 1024,3)]
# %%
# data = np.stack(new_arr, axis=0)
# %%
import numpy as np
from PIL import Image
import os
import shutil
dirs = ['COVID-19', 'NORMAL', 'Viral Pneumonia']

for dir in dirs:
    new_dir = 'data/'+dir+'_imgs'
    for fname in os.listdir('../COVID-img-data/'+dir):
        im = Image.open('../COVID-img-data/'+dir+'/'+fname)
        x = np.asarray(im)
        if x.shape == (1024,1024,3):
            shutil.copy('../COVID-img-data/'+dir+'/'+fname, new_dir)
            
            
            
            
            
            
            
# %%
