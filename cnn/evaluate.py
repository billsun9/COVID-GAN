import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, ImageDataGenerator
import os
model = load_model('models/cnn_v1.h5')

test_data = 'data/test/'
batch_size = 32
'''
def process_data(dir): # returns X and Y
    X = []
    Y = []
    for folder in os.listdir(dir):
        pass
'''    
test_datagen = ImageDataGenerator(
    rescale=1./255
    )

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(256,256),
    batch_size=batch_size,
    color_mode='grayscale'
    )

_, acc = model.evaluate(test_generator)