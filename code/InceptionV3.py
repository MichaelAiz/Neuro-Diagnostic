
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
import os

from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.normalization import BatchNormalization
import TFrecords
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
import itertools
from tensorflow.python.keras.backend import dropout 
import constants



training_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TRAIN, 'train')
val_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_VAL, 'val')
test_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TEST, 'test')



conv_base = keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (440,344,3), pooling = max)
conv_base.trainable = False

model = keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation = 'softmax'))

model.build(input_shape=(8, 440,344,3))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(training_set, epochs = 10, steps_per_epoch=95/8, batch_size=8, validation_data=val_set, validation_steps=19/8)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()






conv_base.trainable = True

for layer in conv_base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False


model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-7)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(training_set, epochs = 70, steps_per_epoch=95/8, batch_size=8, validation_data=val_set, validation_steps=19/8)

model.save('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/NoAugmentNoFineTuneBalanced.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


model.evaluate(test_set)

model = keras.models.load_model('E:/Projects/Neuro-Diagnostic/Models/FineTunedModelNoAugment.h5')


def calculate_auc(model, test_set):

    labels = np.concatenate([labels for images, labels in test_set], axis = 0)
    images = np.concatenate([images for images, labels in test_set], axis = 0)
    n_classes = 3
    y_true = tf.keras.utils.to_categorical(labels, n_classes)
    i = 0
    y_pred = model.predict(x=images)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = skmetrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = skmetrics.auc(fpr[i], tpr[i])
    
    print(type(y_pred))
    # compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = skmetrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = skmetrics.auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 3

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = skmetrics.auc(fpr['macro'], tpr['macro'])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

calculate_auc(model, test_set)
    