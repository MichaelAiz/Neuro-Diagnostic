# Local Constants
import tensorflow as tf


DATABASE = 'E:/Projects/Neuro-Diagnostic/ADNI/Original/'
DB_SUBFOLDERS = ['Complete 2Yr 1.5T/', 'Complete 3Yr 1.5T/', 'Screening_1.5T/']
REGISTERED_DB = 'E:/Projects/Neuro-Diagnostic/ADNI/Registered/'
ATLAS = 'E:/Projects/Neuro-Diagnostic/MNI Atlas/average305_t1_tal_lin.nii'
TFRECORD_PATH = 'E:/Projects/Neuro-Diagnostic/TFRecords/'
SUBFOLDERS = ['CN', 'MCI', 'AD']
TFRECORD_DB = "E:/Projects/Neuro-Diagnostic/ADNI/TFRecords"
TFRECORD_TRAIN = "train.tfrecords"
TFRECORD_TEST = "test.tfrecords"
TFRECORD_VAL = "validation.tfrecords"

# Drive Constants
REGISTERED_DB_DRIVE = ""
CLASS_SUBFOLDERS = ""
TFRECORD_DB_DRIVE = "/content/drive/MyDrive/Neuro-Diagnostic/TFRecords"
TFRECORD_TRAIN_DRIVE = "/content/drive/MyDrive/Neuro-Diagnostic/TFRecords/train.tfrecords"
TFRECORD_TEST_DRIVE = "/content/drive/MyDrive/Neuro-Diagnostic/TFRecords/validation.tfrecords"
TFRECORD_VAL_DRIVE = "/content/drive/MyDrive/Neuro-Diagnostic/TFRecords/test.tfrecords"

# Tensorflow Constants
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
TRAIN_SIZE = 891
VAL_SIZE = 191
TEST_SIZE = 191

# Other constants
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2} 
IMG_SHAPE = (78, 110, 86)
IMG_2D_SHAPE = (440, 344)
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
