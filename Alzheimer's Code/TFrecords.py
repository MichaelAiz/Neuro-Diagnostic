import tensorflow as tf

def _bytes_feature(value):
    if isinstance( value, type(tf.constant(0))): # checks if the passed value is a tensor
        value = value.numpy() # if it is a tensor, converts tensor to numpy ndarray
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


