
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.normalization import BatchNormalization
import TFrecords
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
import itertools
from tensorflow.python.keras.backend import dropout 
import constants


# create TFRecord dataset for training, validation, and testing
training_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TRAIN, 'train')
val_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_VAL, 'val')
test_set = TFrecords.create_tfrecord_dataset(constants.TFRECORD_TEST, 'test')


# load the InceptionV3 network pretrained with ImageNet weights
conv_base = keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (440,344,3), pooling = max)
# freeze the InceptionV3 network
conv_base.trainable = False

model = keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3, activation = 'softmax'))


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# train only the added top layers
history = model.fit(training_set, epochs = 10, steps_per_epoch=95/8, batch_size=8, validation_data=val_set, validation_steps=19/8)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

#unfreeze the InceptionV3 network
conv_base.trainable = True

# freeeze BatchNormalization layers, leaving them unfrozen will cause to large weight updates
for layer in conv_base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.summary()

# retrain the network with the unfrozen base model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-7)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(training_set, epochs = 70, steps_per_epoch=95/8, batch_size=8, validation_data=val_set, validation_steps=19/8)


model.save('E:/Projects/Neuro-Diagnostic/Models/Alzheimers/NoAugmentFineTunedBalancedFinal.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

model.evaluate(test_set)

model = keras.models.load_model('E:/Projects/Neuro-Diagnostic/Models/FineTunedModelNoAugment.h5')

target_names = {
    0: "CN",
    1: "MCI",
    2: "AD"
}

# function to calculate the macro and micro-average ROC and AUC for a given model
def calculate_auc(model, test_set, target_names):

    # generates ndarray of labels and images 
    labels = np.concatenate([labels for images, labels in test_set], axis = 0)
    images = np.concatenate([images for images, labels in test_set], axis = 0)
    n_classes = 3
    # categorically encode labels
    y_true = tf.keras.utils.to_categorical(labels, n_classes)
    y_pred = model.predict(x=images)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # computes fpr and tpr values for each class
    for i in range(3):
        fpr[i], tpr[i], _ = skmetrics.roc_curve(y_true[:, i], y_pred[:, i])
        # computes auc for each class
        roc_auc[i] = skmetrics.auc(fpr[i], tpr[i])
    
    print(type(y_pred))
    # compute micro-average ROC curve and AUC
    # micro-average ROC is computed by aggregating contributiuons from all classes
    # because of the way it is calculated, it is a good metric for imbalanced classes
    fpr['micro'], tpr['micro'], _ = skmetrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = skmetrics.auc(fpr['micro'], tpr['micro'])

    # calculate macro-average ROC and AUC

    # get all fpr values
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # interpolate the ROC curves at the fpr values
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # average the resulting values
    mean_tpr /= 3

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = skmetrics.auc(fpr['macro'], tpr['macro'])

    # plot all ROC curves 
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
        plt.plot(fpr[i], tpr[i], color=color, lw=4, label='ROC curve of class {0} (area = {1:0.2f})'''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Analysis of Final Model')
    plt.legend(loc="lower right")
    plt.show()

#calculate_auc(model, test_set, target_names)
    