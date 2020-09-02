import numpy as np
import os
from keras.models import load_model

model = load_model('models/cnn_v1.h5')
# %%
normal_imgs = os.listdir('../COVID-img-data/NORMAL')
normal_img1 = normal_imgs[0]
normal_img2 = normal_imgs[1]
normal_img3 = normal_imgs[2]
normal_img4 = normal_imgs[3]
# %%
from keras.preprocessing.image import load_img
img1 = load_img('../COVID-img-data/NORMAL'+'/'+normal_img1)
img2 = load_img('../COVID-img-data/NORMAL'+'/'+normal_img2)
img3 = load_img('../COVID-img-data/NORMAL'+'/'+normal_img3)
img4 = load_img('../COVID-img-data/NORMAL'+'/'+normal_img4)

#%%
def process(img):
    x = img.resize((256,256))
    x = np.array(x)
    x = np.mean(x, axis=2) # from rgb to grayscale
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')
    x /= 255.0
    return x
# %%
process1 = process(img1)
process2 = process(img2)
process3 = process(img3)
process4 = process(img4)
# %%
pred = model.predict(process1)
print(model.predict(process2))
print(model.predict(process3))
print(model.predict(process4))