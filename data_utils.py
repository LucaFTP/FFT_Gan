import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import fftpack
from skimage.transform import rotate, resize

def create_folders(redshift):
    
    version='_z_'+str(redshift)
    CKPT_OUTPUT_PATH = '/leonardo_scratch/fast/INA24_C3B13/FFT_GAN_ckpts'+version+'_4_final'
    IMG_OUTPUT_PATH = 'FFT_GAN_Images'+version
    ARCH_OUTPUT_PATH = 'FFT_GAN_Arch'+version
    LOSS_OUTPUT_PATH = 'FFT_GAN_Loss'+version
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
        
def polar_inv_fft(r, phi):
    z = r * np.exp(1j * phi)
    return abs(fftpack.ifft2(z))

def process_with_mask(img:np.ndarray, target_size, mask_parameter:int):
    '''
    Given the form of the data at the __get_input stage,
    this function reshapes the arrays and applies the FFT. After that, it applies
    a mask that cuts out high frequencies.

    It returns the blured imaged reconstructed by applying the IFFT to the masked uv-plane.
    
    img         :       np.ndarray      Input image
    target_size :       iterable object (tuple, list or array)        Desired pixel size of the images
    mask_parameter :    int        Variable that regulates the extension of the step-function mask
    '''
    x, y = np.linspace(0, *target_size), np.linspace(0, *target_size)
    xv, yv = np.meshgrid(x, y)

    center = (target_size[0] // 2, target_size[1] // 2)
    inf_x, inf_y = center[1] - mask_parameter, center[0] - mask_parameter
    sup_x, sup_y = center[1] + mask_parameter, center[0] + mask_parameter
    
    mask = np.zeros(target_size)
    mask[(xv >= inf_x) & (xv <= sup_x) & (inf_y >= inf_y) & (yv <= sup_y)] = 1

    img = resize(np.squeeze(img), target_size)
    image_ft = np.fft.fftshift(fftpack.fft2(img))
    masked_ft_module, ft_phase = np.absolute(image_ft) * mask, np.angle(image_ft)

    return polar_inv_fft(masked_ft_module, ft_phase)

def load_meta_data(redshift, show=False):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()#.sort_values(by=['mass', 'rot']).reset_index(drop=True)
    
    # Showing what all is in my data
    if show:
        get_unique(meta_data)
    
    return meta_data

        
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, meta_data, X_col, y_col, batch_size, target_size, rot_col=False, shuffle=True):
        
        self.meta_data = meta_data.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.rot_col = rot_col
        self.n = len(self.meta_data)
        self.data_dir = "/leonardo_scratch/fast/INA24_C3B13/ALL_ROT_npy_version/1024x1024/"

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, target_size, rotation_angle=0):

        file_name = img_id + '.npy'
        file_name = self.data_dir + file_name
            
        img = np.load(file_name).astype('float32')
        img = img[256:768, 256:768]
        img = tf.image.resize(np.expand_dims(img, axis=-1), target_size).numpy()
        
        return (img - np.min(img))/(np.max(img) + np.min(img))
    
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

class SmoothingCustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, meta_data, X_col, y_col, batch_size, target_size, initial_mask_par=4, rot_col=False, shuffle=True):
        
        self.meta_data = meta_data.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.mask_par = initial_mask_par
        self.shuffle = shuffle
        self.rot_col = rot_col
        self.n = len(self.meta_data)
        self.data_dir = "/leonardo_scratch/fast/INA24_C3B13/ALL_ROT_npy_version/1024x1024/"

    def on_epoch_end(self):
        if (self.mask_par < (self.target_size[0]//2)):
            self.mask_par += 1
        if self.shuffle:
            print('Shuffling the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, target_size, rotation_angle=0):

        file_name = img_id + '.npy'
        file_name = self.data_dir + file_name
            
        img = np.load(file_name).astype('float32')
        img = img[256:768, 256:768]
        img = (img - np.min(img))/(np.max(img) + np.min(img))

        blured_image = process_with_mask(img, target_size, self.mask_par)
        
        return tf.image.resize(np.expand_dims(blured_image, axis=-1), target_size).numpy()
    
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
