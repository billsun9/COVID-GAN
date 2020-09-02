import os
from PIL import Image
# %%
dir = '../COVID-img-data/Viral Pneumonia'
fnames = os.listdir(dir)[:5]
# %%
for fname in fnames:
    im = Image.open(dir+'/'+fname)
    im.show()