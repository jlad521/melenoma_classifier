#init-file 

#define .filepaths
#load libraries
#load flags and appropriate files?


.filepaths=list(
  dataset_root =  file.path('/media/udaman/Hard Disk/melenoma_ds'),
  train_jpgs = file.path(dataset_root, 'jpeg/train'),
  test_jpgs  =  file.path(dataset_root, 'jpeg/test'),
  train_dcim = file.path(dataset_root, 'train'),
  test_dcim  = file.path(dataset_root, 'test'),
  dev_wd = file.path('/home/udaman/Documents/isic')
)

library(tensorflow)
library(tidyverse)
library(magrittr)
library(tfdatasets)

#enable gpu growth 
#source any files in init
