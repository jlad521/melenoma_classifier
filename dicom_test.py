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


 * Train/Dev split (test set) ensuring equal distribution of cancerous tumors
 * 
'''

from model import build_model
from dataset import ds #import dataset (should be train/dev sets)

model = build_model()

history = model.fit(
      ds,
      steps_per_epoch=10,  
      epochs=10,
      verbose=1)
print(history)

