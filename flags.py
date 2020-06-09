#flags.py
import os

ds_r = '/media/udaman/Hard Disk/melenoma_ds'
flags = {
    'ds_root':ds_r,
    'train_dcim':os.path.join(ds_r, 'train'),
    'test_dcim':os.path.join(ds_r, 'test'),
    'train_jpg':os.path.join(ds_r, 'jpegs/train'),
    'test_jpg':os.path.join(ds_r, 'jpegs/test'),
    'dev_wd':'/home/udaman/Documents/isic',
    'input_h': 224,
    'input_w': 224,
    'batch_size':16
}