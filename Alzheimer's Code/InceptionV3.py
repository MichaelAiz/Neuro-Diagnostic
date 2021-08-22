import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
import os
import TFrecords
import matplotlib.pyplot as plt

from tensorflow.python.keras.backend import dropout 
import constants

# training_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TRAIN, 'train')
# val_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_VAL, 'val')
# test_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TEST, 'test')


# conv_base = keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (440, 344, 3), pooling = max)
# conv_base.trainable = False

# model = keras.models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.8))
# model.add(layers.Dense(3, activation = 'softmax'))

# model.summary()


# optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-3)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# history = model.fit(training_set, epochs = 10, steps_per_epoch=891/8, batch_size=8, validation_data=val_set)
# model.save('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/model.h5')

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

# model.save_weights('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/model.h5')
# model_json = model.to_json()
# with open('model.json', "w") as json_file:
#     json_file.write(model_json)
# json_file.close()




# # for layer in model.layers:
# #     if not (isinstance(layer, layers.BatchNormalization)):
# #         layer.trainable = True

# for layer in model.layers:
#     layer.trainable = True

# optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-7)
# history = model.fit(training_set, epochs = 70, steps_per_epoch=891/8, batch_size=8, validation_data=val_set)

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')



# model = keras.models.load_model('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/model.h5')
# model.evaluate(test_set)


    