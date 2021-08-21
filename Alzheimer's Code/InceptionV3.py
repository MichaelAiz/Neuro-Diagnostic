import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
import os
import TFrecords

from tensorflow.python.keras.backend import dropout 
import constants

training_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TRAIN_DRIVE, 'train')
val_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_VAL_DRIVE, 'val')

conv_base = keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (440, 344, 3), pooling = max)
conv_base.trainable = False

model = keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(3, activation = 'softmax'))

model.summary()


optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['acc'])

model.fit(training_set, epochs = 10, steps_per_epoch=891/8, batch_size=8, validation_data=val_set)


# for layer in model.layers:
#     if not (isinstance(layer, layers.BatchNormalization)):
#         layer.trainable = True

for layer in model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-7)
model.fit(training_set, epochs = 70, steps_per_epoch=891/8, batch_size=8, validation_data=val_set)
    