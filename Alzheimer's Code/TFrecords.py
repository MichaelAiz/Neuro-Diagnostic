import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tensorflow.python.ops.gen_parsing_ops import serialize_tensor
import constants
import os
import preprocessing
from sklearn.model_selection import train_test_split
import IPython
from IPython.display import display
import matplotlib.pyplot as plt

# functions to convert tensorflow types to a tf.train.Example compatible Feature


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # checks if the passed value is a tensor
        value = value.numpy()  # if it is a tensor, converts tensor to numpy ndarray
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


# create destination paths for TFrecords
tfrecord_train = os.path.join(
    constants.TFRECORD_PATH, constants.TFRECORD_TRAIN)
tfrecord_val = os.path.join(constants.TFRECORD_PATH, constants.TFRECORD_VAL)
tfrecord_test = os.path.join(constants.TFRECORD_PATH, constants.TFRECORD_TEST)

# writes all files in filenames to TFrecord


def write_to_tf_record(filenames, record_name):
    writer = tf.io.TFRecordWriter(record_name)
    for file in filenames:
        img, label = preprocessing.load_2D_image(file)
        print(label)

        # create feature mapping for the Example
        features = {
            'label': _int64_feature(label),
            'height': _int64_feature(img.shape[1]),
            'width': _int64_feature(img.shape[0]),
            'depth': _int64_feature(img.shape[2]),
            'image_raw': _bytes_feature(serialize_array(img))
        }

        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())

    writer.close()

# parse a single TFRecord example


def parse_tfr_example(element):
    # create a dictionary describing the features
    example_features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(element, example_features)

    # extract the features
    label = example['label']
    img_raw = example['image_raw']
    height = example['height']
    width = example['width']
    depth = example['depth']

    # reconstruct the original image
    image = tf.io.parse_tensor(img_raw, out_type=tf.float64)
    image = tf.reshape(image, shape=[height, width, depth])

    return (image, label)

# create a dataset from the TFRecords
def create_tfrecord_dataset(filename, set_type):
    raw_dataset = tf.data.TFRecordDataset(filename)
    # parse every example in the dataset
    # let the runtime decide on optimal number of parallel calls
    dataset = raw_dataset.map(
        parse_tfr_example, num_parallel_calls=constants.AUTOTUNE)
    # set the batch size, this is the number of samples the model will process before making adjustments
    dataset = dataset.batch(constants.BATCH_SIZE)
    # let runtime decide on optimal prefetch to increase performance
    dataset = dataset.prefetch(buffer_size=constants.AUTOTUNE)
    dataset = dataset.repeat() if set_type == 'train' else dataset
    return dataset


filenames = np.array([])  # array to store all registered images


# add all files from folders to filenames array
for path, dirs, files in os.walk(constants.REGISTERED_DB):
    for file in files:
        fixed_file_path = path + "/"
        filenames = np.append(filenames, os.path.join(fixed_file_path, file))


# split the filenames array into training, validation, and testing sets
# 70% of images go to training set, 15% to both validation and testing




def write_tf_records(filenames):
    train_set, test_set = train_test_split(filenames, train_size=0.70)
    test_set, val_set = train_test_split(test_set, train_size=0.5)
    write_to_tf_record(train_set, tfrecord_train)
    write_to_tf_record(val_set, tfrecord_val)
    write_to_tf_record(test_set, tfrecord_test)

