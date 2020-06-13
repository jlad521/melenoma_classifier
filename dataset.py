#dataset.py file
'''
loads the dataset from disk, returning a train and validation ds
'''

import pydicom
import numpy as np 
import sys
import glob
import pandas as pd
import tensorflow as tf
import os 
import csv
from collections import defaultdict 
from flags import flags
from matplotlib import pyplot as plt

from utils import *

def process_file(path):
    '''
    takes a path to single jpg file, returns tf operations to open, and preprocesses.

    note: i don't think these ops are on the graph because it's being called with from_generator?

    TODO: add some image augmentation (rotation, flips, etc)
    '''
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.central_crop(img, 0.5)
    img = tf.image.resize(img, [flags['input_h'], flags['input_w']])
    
    return tf.image.convert_image_dtype(img, tf.float32) / 255. #convert images to [0,1]
    

def train_gen():
    for img in os.listdir(flags['train_jpg']):
        img_name = img.split('.')[0]
        gt = np.asarray(train_dict[img_name]['gt'], dtype=np.int16)
        tf_img = process_file(os.path.join(flags['train_jpg'],img))
        #soon, use other info including age, location, etc into the MLP at the end 
        yield (tf_img, gt)

def make_gen_ds():
    ds = tf.data.Dataset.from_generator(
        train_gen, 
        output_types= (tf.float32, tf.int16),
        output_shapes= (tf.TensorShape(flags['input_shape']), tf.TensorShape([])))

    return ds.repeat().shuffle(128).batch(flags['batch_size']).prefetch(1)


train_df, anot_dict, diagnosis_dict, sex_lookup = get_csv_dicts()
train_dict = csv_to_dict(os.path.join(flags['ds_root'], 'train.csv'), anot_dict, diagnosis_dict,sex_lookup)

ds = make_gen_ds()
#print(ds)