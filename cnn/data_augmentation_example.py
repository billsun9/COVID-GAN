from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# %%
datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8,1.2],
    zoom_range=[0.9, 1.0]
    )

img = load_img('../COVID-img-data/COVID-19/COVID-19 (1).png')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
# %% flow --> uses img already in memory
# flow command generates batches of randomly transformed images
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='augmented_imgs/',
                          save_prefix='xray', save_format='jpeg'):
    i += 1
    if i>8:
        break # otherwise generator loops infinitely
# %%     
# flow_from_directory takes imgs from the raw_imgs folder
i = 0
for batch in datagen.flow_from_directory('raw_imgs/train/', batch_size=1, save_to_dir='augmented_imgs2/',
                          save_prefix='xray', save_format='png', color_mode='grayscale'):
    i += 1
    if i>8:
        break # otherwise generator loops infinitely     
# %%
import os
dir = 'augmented_imgs'
for fname in os.listdir(dir):
    os.remove(dir+'/'+fname)