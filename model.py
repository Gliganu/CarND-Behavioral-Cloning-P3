import numpy as np
import os
import pandas as pd
import PIL
import PIL.Image
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import cv2

from keras.models import Sequential, Model
from keras.layers import Input, merge, Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.callbacks import CSVLogger, ModelCheckpoint

def shorten(df):
    collist = [col for col in df.columns if col not in ['img']]
    return df[collist]

def explode_df(data_df):
    exploded_df = []
    
    for col in ['center','left','right']:
        partial_df = data_df[[col] + ['steering','throttle','speed']]
        
        partial_df=partial_df.rename(columns = {col:'img_path'})
        
        exploded_df.append(partial_df)
        
    exploded_df = pd.concat(exploded_df)
    return exploded_df

def change_col_path(data_df, col_name):
    data_df.loc[:,col_name] = data_df.loc[:,col_name].apply(lambda name: data_path + "IMG/" + os.path.basename(name))
    return data_df

def read_imgs(data_df):
    data_df.loc[:,'img'] = data_df.loc[:,'img_path'].progress_apply(lambda path: read_image(path))
    return data_df

def read_df(csv_path, nr_elems = None):
    data_df = pd.read_csv(csv_path,  names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    
    if(nr_elems != None):
        data_df = data_df.iloc[:nr_elems]
        
    data_df = explode_df(data_df)    
    data_df = change_col_path(data_df, 'img_path')
    
    data_df = read_imgs(data_df)
    
    data_df = data_df.set_index('img_path')
    
    data_df = shuffle(data_df, random_state = 0)
    return data_df

def crop(image):
    return image[60:-25, :, :]

def read_image(path):
    img = PIL.Image.open(path)
    img = np.asarray(img)
    img = crop(img)
    
    return img

def split_train_test(data_df, train_percentage):
    
    nr_train = int(train_percentage * len(data_df))
    train_df = data_df[:nr_train]
    test_df = data_df[nr_train:]
    
    return train_df, test_df

def get_formated_data(data_df):
    imgs = np.stack(data_df['img'].tolist())
    labels = np.stack(data_df['steering'].tolist())
    
    imgs = np.transpose(imgs,(0,3,1,2))
    return imgs, labels

def get_flipped_df(data_df):
    
    augmented_df = data_df.copy()
    
    augmented_df.loc[:,'img'] = augmented_df.loc[:,'img'].apply(lambda img: cv2.flip(img, 1))
    augmented_df.loc[:,'steering'] = augmented_df.loc[:,'steering'].apply(lambda steering: -steering)
    
    return augmented_df

def get_model():
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(3,) + img_size, output_shape = (3,) + img_size))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization(axis=1))
    
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(BatchNormalization(axis=1))
    
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(BatchNormalization(axis=1))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    
    return model

if __name__ == '__main__':
    
    #set paths
    data_path = "../../datasets/self-driving-car/"
    img_path = data_path + "IMG/"
    csv_path = data_path + "driving_log.csv"
    img_size = (75, 320)
    
    #read data
    data_df = read_df(csv_path)
    
    #augment data
    augmented_df = get_flipped_df(data_df)
    data_df = pd.concat([data_df,augmented_df])
    data_df = shuffle(data_df,random_state = 0)
    
    #split in trian/test
    train_df, test_df = split_train_test(data_df, train_percentage = 0.8)
    
    train_imgs, train_labels = get_formated_data(train_df)
    test_imgs, test_labels = get_formated_data(test_df)
    
    #train model
    model = get_model()
    
    model.fit(train_imgs, train_labels,
              nb_epoch= 100,               
              validation_data = (test_imgs, test_labels),
              callbacks = [CSVLogger("./training.txt"), \
                          ModelCheckpoint("./model_temp_2.h5", monitor='val_loss', verbose= 1, save_best_only=True, mode='min')
                 ]
      )
    
    
    
    
    
    
    
    
    
    
    