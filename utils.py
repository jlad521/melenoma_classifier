#utils file
'''
holds utility functions that may be used in various classes
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

def get_metadata(row, sex_lookup, anot_sites_dict, diagnosis_dict):
    cur_meta = [sex_lookup[row[2]]] 
    if row[3] == '': row[3] = 42 #should be average age? or look into data cleansing effort
    cur_meta.append(int(float(row[3])))
    cur_meta.append(anot_sites_dict[row[4]])
    cur_meta.append(diagnosis_dict[row[5]])
    gt = int(row[-1])
    return gt, cur_meta

def csv_to_dict(csv_file, anot_sites_dict, diagnosis_dict, sex_lookup):
    df = defaultdict()

    with open(csv_file) as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        next(csv_reader) #skip header line
        for row in csv_reader:
            gt, cur_meta = get_metadata(row, sex_lookup, anot_sites_dict, diagnosis_dict)
            df[row[0]] = {'gt':gt, 'meta':cur_meta}
  
    return df 


def get_csv_dicts():
    '''
    Searching pandas dataframe for each patient ID key is very expensive.
    Instead, go through the csv file containing all metadata and label once, 
    placing a {'gt', 'meta'} value in the `csv_lookup` dictionary (patient ID is the key) 

    @return csv_lookup: {'patient_id':{'gt':np.array[1], 'meta':[np.array]}}
    @return anot_sites_dict: converting metadata strings to int labels
    @return diagnosis_dict: converting metadata strings to int labels
    '''
    csv_lookup = pd.read_csv(os.path.join(flags['ds_root'], 'train.csv'))

    unique_anot_sites = csv_lookup['anatom_site_general_challenge'].unique()
    anot_sites_dict = {str(name):i+1 for i, name in enumerate(unique_anot_sites)}
    anot_sites_dict['']=1

    unique_diagnosis_sites = csv_lookup['diagnosis'].unique()
    diagnosis_dict = {str(name):i for i, name in enumerate(unique_diagnosis_sites)}
    
    sex_lookup = {'male':0, 'female':1, '':2}

    return csv_lookup, anot_sites_dict, diagnosis_dict, sex_lookup


def read_dicom_ds(ds_dir):
    for curf in os.listdir(ds_dir):
        try:
            cur_dcm = pydicom.dcmread(os.path.join(ds_dir,curf), force=True)
        except:
            print('skipped one')
            continue
        #at some point, extract metadata, or extra info about the patient
        print(cur_dcm)  
     

def test_ds(ds):
    it = iter(ds)
    b = next(it)
    print(b)



def slice_single_element_4d(tensor, size = [], batch_pos=0):
    sliced = tf.slice(tensor, [batch_pos,0,0,0], size)
    return tf.reshape(sliced, flags['input_shape'] )

def visualize_ds(ds):
    it = iter(ds)
    for img in it:
        for batch_pos in range(flags['batch_size']):
            cur_img = slice_single_element_4d(img[0], [1,flags['input_h'], flags['input_w'],3], batch_pos).numpy()
            cur_label = img[1].numpy()
            plt.imshow(cur_img)
            plt.title(cur_label[batch_pos])
            plt.show()



      ### LOAD IMAGE GENERATOR  ### 
def image_generator():
      from tensorflow.keras.preprocessing.image import ImageDataGenerator

      # All images will be rescaled by 1./255
      train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

      validation_datagen = ImageDataGenerator(rescale=1/255)

      train_generator = train_datagen.flow_from_directory(
            flags['train_jpg'],  # This is the source directory for training images
            target_size=(300, 300),  # All images will be resized to 150x150
            batch_size=32,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

      validation_generator = validation_datagen.flow_from_directory(
            flags['test_jpg'],  # This is the source directory for training images
            target_size=(300, 300),  # All images will be resized to 150x150
            batch_size=32,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')        
      return train_datagen, validation_datagen