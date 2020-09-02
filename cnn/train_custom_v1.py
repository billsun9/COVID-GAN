import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import BatchNormalization
# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8,1.2],
    zoom_range=[0.9, 1.0]
    )

test_datagen = ImageDataGenerator(
    rescale=1./255
    )
# this is a data generator that will read pictures found in 'data/train' and indefinitely generate
# batches of augmented image data

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(256,256),
    batch_size=batch_size,
    color_mode='grayscale'
    )

validation_generator = test_datagen.flow_from_directory(
    'data/val',
    target_size=(256,256),
    batch_size=batch_size,
    color_mode='grayscale'
    )
# %%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
import time
def define_model(input_shape=(256,256,1)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
t1 = time.time()
hist = model.fit_generator(train_generator,
        steps_per_epoch=2093 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=522 // batch_size)
print('train time: '+str(time.time()-t1))
#model.save('models/cnn_v1.h5')
model.save('models/cnn_v1_retest.h5')
# %%
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()