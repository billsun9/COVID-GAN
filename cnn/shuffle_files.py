# moves img data into train validation split
import os
import numpy as np
import random
import shutil
# %%
train_path = 'data/train/'
val_path = 'data/val/'
test_path = 'data/test/'

data_folders = ['COVID-19/', 'NORMAL/', 'Viral Pneumonia/']

test_ratio = 0.1 # 10% of data is used for testing
val_ratio = 0.2 # 20% of training data is used for validation

for folder_name in data_folders:
    dir_path = '../COVID-img-data/'+folder_name
    img_paths = os.listdir(dir_path)
    
    random.shuffle(img_paths)
    
    test_split = round(test_ratio * len(img_paths))
    testing_data = img_paths[:test_split]
    training = img_paths[test_split:]
    
    random.shuffle(training)
    val_split = round(val_ratio * len(training))
    validation_data = training[:val_split]
    training_data = training[val_split:]
    
    for img_path in training_data:
        shutil.copy(dir_path+img_path, train_path+folder_name)
    for img_path in testing_data:
        shutil.copy(dir_path+img_path, test_path+folder_name)
    for img_path in validation_data:
        shutil.copy(dir_path+img_path, val_path+folder_name)
# %%
for idx in range(len(training)):
    if training[idx] in new_list:
        print('has dups')
        break
    new_list.append(training[idx])