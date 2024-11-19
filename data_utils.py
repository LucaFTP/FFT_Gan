import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize, rotate
import tensorflow as tf

def apply_floor_log(array):
    array[array <= 0] = np.min(array[array > 0])
    return np.log(array)

def create_folders():
    
    version='_z_0.03'
    CKPT_OUTPUT_PATH = 'VAC-PGGAN_ckpts'+version
    IMG_OUTPUT_PATH = 'VAC-PGGAN_Images'+version
    ARCH_OUTPUT_PATH = 'VAC-PGGAN_Arch'+version
    LOSS_OUTPUT_PATH = 'VAC-PGGAN_Loss'+version
    DATASET_OUTPUT_PATH = 'synthetic_data'+version

    try:
        os.mkdir(CKPT_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(IMG_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(ARCH_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(LOSS_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(DATASET_OUTPUT_PATH)
    except FileExistsError:
        pass
    
    return CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, ARCH_OUTPUT_PATH, LOSS_OUTPUT_PATH, DATASET_OUTPUT_PATH

def augment_by_rotation(data, angles=[90,180,270]):
    original_data = data.copy()
    for angle in angles:
        original_data['rotation'] = angle
        data = pd.concat([data,original_data])
    return data

def get_unique(data):
    for col in data.columns[1:]:
        print(f'\n"{col}" has {len(data[col].unique())} unique values: {data[col].unique()}')
        
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, meta_data, X_col, y_col, batch_size, target_size, freqs=['fixed'], rot_col=False, blur = 0, shuffle=True):
        
        self.meta_data = meta_data.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.freqs = freqs
        self.blur = blur
        self.rot_col = rot_col
        self.n = len(self.meta_data)
        self.data_dir = "/leonardo/home/userexternal/lfontana/ALL_ROT_npy_version/1024x1024/"

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, target_size, rotation_angle=0):

        imgs = {}

        for freq in self.freqs:
            
            file_name = img_id + '.npy'
            file_name = self.data_dir + file_name
            
            imgs[freq] = np.load(file_name).astype('float32')

            if self.rot_col:
                imgs[freq] = rotate(imgs[freq], rotation_angle)

            imgs[freq] = (imgs[freq] - np.mean(imgs[freq]))/(imgs[freq] + np.mean(imgs[freq]))
            # imgs[freq] = (imgs[freq] - np.min(imgs[freq]))/(np.max(imgs[freq]) + np.min(imgs[freq]))
            # imgs[freq] = apply_floor_log(imgs[freq])
            
        stacked_img = np.stack(list(imgs.values()), axis=2)
        image_arr = tf.image.resize(stacked_img,(target_size[0], target_size[1])).numpy()       
        
        return image_arr
    
    def __get_output(self, label):
        return label
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_col_batch = batches[self.X_col]

        if self.rot_col:
            rot_col_batch = batches[self.rot_col]
            X_batch = np.asarray([self.__get_input(x, self.target_size, rot) for (x,rot) in zip(X_col_batch,rot_col_batch)])
        else: 
            X_batch = np.asarray([self.__get_input(x, self.target_size) for x in X_col_batch])
            
        y_col_batch = batches[self.y_col]
        
        y_batch = np.asarray([self.__get_output(y) for y in y_col_batch])
        
        return X_batch, y_batch
    
    def __getitem__(self, index):
        
        # The role of __getitem__ method is to generate one batch of data. 
        
        meta_data_batch = self.meta_data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(meta_data_batch)
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

def load_meta_data(redshift, print_opt=False):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()#.sort_values(by=['mass', 'rot']).reset_index(drop=True)
    if (print_opt==True):
        print(f"Data Shape: {meta_data.shape}")
    '''
    aug_meta_data = augment_by_rotation(meta_data)
    print(f"Data Shape of augmented dataset: {aug_meta_data.shape}")
    '''
    # Showing what all is in my data
    get_unique(meta_data)
    
    return meta_data