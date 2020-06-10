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
'''
TODO: 
 * read images from dicom format?  for now using jpeg and reading csv doc for the info
 * get dicom dataset working
 * do visualizations of the data after the crop and preprocessing
    * visualize all the cancerous images 
 * 

'''

def read_dicom_ds(ds_dir):
    for curf in os.listdir(ds_dir):
        try:
            cur_dcm = pydicom.dcmread(os.path.join(ds_dir,curf), force=True)
        except:
            print('skipped one')
            continue
        #at some point, extract metadata, or extra info about the patient
        print(cur_dcm)
        d = cur_dcm.to_json_dict()
        print(cur_dcm)
        #print(d.keys)
        #print(cur_dcm['Planar Configuration'])
        #image = cur_dcm.pixel_array
        #shape = image.shape
        #res = np.reshape(image, (shape[0] //3, shapep[1]//3))
        #print(image)

#read_dicom_ds(flags['train_dcim'])

#ds = pydicom.dcmread('/home/udaman/Documents/ISIC_0015719.dcm')
def get_metadata(row, sex_lookup, anot_sites_dict, diagnosis_dict):
    cur_meta = [sex_lookup[row[2]]] 
    if row[3] == '': row[3] = 42 #should be average age? or look into data cleansing effort
    cur_meta.append(int(float(row[3])))
    cur_meta.append(anot_sites_dict[row[4]])
    cur_meta.append(diagnosis_dict[row[5]])
    gt = int(row[-1])
    return gt, cur_meta

def gt_lookup(csv_file, anot_sites_dict, diagnosis_dict):
    df = defaultdict()
    sex_lookup = {'male':0, 'female':1, '':2}
    #anatom_site_lookup = {unique}
    with open(csv_file) as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            gt, cur_meta = get_metadata(row, sex_lookup, anot_sites_dict, diagnosis_dict)
            df[row[0]] = {'gt':gt, 'meta':cur_meta}
  
    return df 


def train_csv_preprocessing():
    train_df = pd.read_csv(os.path.join(flags['ds_root'], 'train.csv'))
    unique_anot_sites = train_df['anatom_site_general_challenge'].unique()
    anot_sites_dict = {str(name):i+1 for i, name in enumerate(unique_anot_sites)}
    anot_sites_dict['']=1

    unique_diagnosis_sites = train_df['diagnosis'].unique()
    diagnosis_dict = {str(name):i for i, name in enumerate(unique_diagnosis_sites)}
    #train_df.map(lambda x: )
    return train_df, anot_sites_dict, diagnosis_dict

train_df, anot_dict, diagnosis_dict = train_csv_preprocessing()
train_dict = gt_lookup(os.path.join(flags['ds_root'], 'train.csv'), anot_dict, diagnosis_dict)


#train_df = train_csv_preprocessing()
def process_file(path):

    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.central_crop(img, 0.5)
    img = tf.image.resize(img, [flags['input_h'], flags['input_w']])
    
    return tf.image.convert_image_dtype(img, tf.float32) / 255.
    
def train_gen():
    for img in os.listdir(flags['train_jpg']):
        img_name = img.split('.')[0]
        gt = np.asarray(train_dict[img_name]['gt'], dtype=np.int16)
        tf_img = process_file(os.path.join(flags['train_jpg'],img))

        #soon, use other info including age, location, etc into the MLP at the end 
        yield (tf_img, gt)

print('about to launch train_gen')


def make_gen_ds():
    ds = tf.data.Dataset.from_generator(
        train_gen, 
        output_types= (tf.float32, tf.int16),
        output_shapes= (tf.TensorShape(flags['input_shape']), tf.TensorShape([])))

    ds = ds.shuffle(128).batch(flags['batch_size']).prefetch(1)
    return ds


def make_ds():
    ds = tf.data.Dataset.list_files(flags['train_jpg']+'/*')
    ds = ds.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.shuffle(buffer_size=64)
    ds = ds.repeat()
    ds = ds.batch(flags['batch_size'])
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    print(ds)
    print('done')
    return ds

ds = make_gen_ds()

def test_train_gen():
    for x in train_gen():
        print(x)

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

#visualize_ds(ds)

from model import build_model
model = build_model()

history = model.fit(
      ds,
      steps_per_epoch=10,  
      epochs=10,
      verbose=1)
print(history)

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