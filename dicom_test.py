import pydicom
import numpy as np 
import sys
import glob
import pandas as pd
#import tensorflow as tf
import os 

from flags import flags
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

read_dicom_ds(flags['train_dcim'])

#ds = pydicom.dcmread('/home/udaman/Documents/ISIC_0015719.dcm')


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
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=1e-4),
                metrics=['accuracy'])
    return model



def train_csv_preprocessing():
    train_df = pd.read_csv(os.path.join(flags['ds_root'], 'train.csv'))
    unique_anot_sites = train_df['anatom_site_general_challenge'].unique()
    anot_sites_dict = {name:i for name, i in enumerate(unique_anot_sites)}
    return train_df

train_df = train_csv_preprocessing()


def train_gen():
    for img in os.listdir(flags['train_jpg']):
        entry = train_df[train_df['image_name']==img.split('.')[0]]
        gt = entry['target'].astype(np.int16)
        tf_img = process_file(os.path.join(flags['ds_root'],img))
        #soon, use other info including age, location, etc into the MLP at the end 
        #print((tf_img, gt))
        yield (tf_img, gt) 

def make_gen_ds():
    ds = tf.data.Dataset.from_generator(
        train_gen, 
        ([tf.float32, tf.float32, tf.float32], tf.int16),
        (tf.TensorShape([flags['input_h'], flags['input_w'], 3]), tf.TensorShape([1])))
    print(ds)

def get_label(path):
    substrings = tf.strings.split(path, os.path.sep)
    entry = train_df[train_df['image_name']==tf.strings.split(substrings[-1],sep=('.'))[0]].values
    return tf.cast(entry[0][7], tf.int16)
    #gt = entry['target'].astype(tf.int16)
    #return gt
    
#test = get_label(os.path.join(flags['train_jpg'], 'ISIC_0068279.jpg' ))
#print(test)
def process_file(path):

    gt = get_label(path)
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    
    return tf.image.convert_image_dtype(img, tf.float32), gt
    #return tf.image.resize(img, [flags['input_h'], flags['input_w']]) Want to do cropping instead of resizing

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

def test_ds(ds):
    it = iter(ds)
    b = next(it)
    print(b)

ds = make_ds()
test_ds(ds)

#train_gen()
# history = model.fit(
#       train_gen,
#       steps_per_epoch=100,  
#       epochs=100,
#       verbose=1)
      #validation_data = validation_generator,
      #validation_steps=8)