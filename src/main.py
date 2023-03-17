import numpy as np
import pandas as pd

import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale

import os
from pathlib import Path
import re

import tensorflow as tf

#train_dir = 'src\images'
train_dir = 'images'
train_path = Path(train_dir)
files = list (train_path.glob('*.png'))
files
names = [os.path.split(x)[1] for x in list(train_path.glob('*.png'))]

image_df= pd.concat ([pd.Series(names, name = 'Name'), pd.Series(files, name ='Filepath').astype(str) ], axis= 1)

image_df['Name'] = image_df ['Name'].apply( lambda x: re.sub (r'\.\w+$','',x))

df = pd.read_csv ('pokemon.csv')
#df = pd.read_csv ('src\pokemon.csv')
df

train_df = image_df.merge(df, on= 'Name')
train_df

train_df = train_df.drop('Type2', axis=1)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split =0.2,
    rescale = 1./255

)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col= 'Filepath',
    y_col= 'Type1',
    target_size= (120,120),
    color_node = 'rgba',
    class_node ='sparse',
    batch_size = 32,
    shuffle= True,
    seed= 1,
    subset = 'training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col= 'Filepath',
    y_col= 'Type1',
    target_size= (120,120),
    color_node = 'rgba',
    class_node ='sparse',
    batch_size = 32,
    shuffle= True,
    seed= 1,
    subset = 'validation'


)
     
inputs = tf.keras.Input(shape=(120,120,3))

conv1= tf.keras.layers.Conv2D (filters =64, kernel_size=(8,8), activation='relu')(inputs)
pool1 = tf.keras.layers.MaxPool2D()(conv1)

conv2= tf.keras.layers.Conv2D (filters =128, kernel_size=(8,8), activation='relu')(pool1)
pool2 = tf.keras.layers.MaxPool2D()(conv2)

conv3= tf.keras.layers.Conv2D (filters =256, kernel_size=(8,8), activation='relu')(pool2)
pool3 = tf.keras.layers.MaxPool2D()(conv3)

outputs =tf.keras.layers.GlobalAveragePooling2D()(pool3)



feature_extractor = tf.keras.Model(inputs=inputs, outputs= outputs)
feature_extractor.summary()

clf_inputs = feature_extractor.input
clf_outputs = tf.keras.layers.Dense(units= 1, activation= 'sigmoid')(feature_extractor.output)

classifier = tf.keras.Model(inputs= clf_inputs, outputs= clf_outputs)

classifier.summary()

classifier.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
history = classifier.fit(
    train_data,
    validation_data = val_data,
    batch_size = 32,
    epochs= 7,
    callbacks = [
                 tf.keras.callbacks.EarlyStopping(
                     monitor= 'val_loss',
                     patience = 5,
                     restore_best_weights = True
                 ),
                 tf.keras.callbacks.ReduceLROnPlateau()
    ]
)

feature_extractor.layers

print(feature_extractor.layers[1].weights)

print("*********End******")