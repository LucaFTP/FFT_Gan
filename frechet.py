import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.linalg import sqrtm
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    
    return np.asarray(images_list)
    
# calculate frechet inception distance
def calculate_fid(images1, images2):
    # Define the model and preprocess the images
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    images1, images2 = preprocess_input(images1), preprocess_input(images2)
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
         covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
 
    return fid
    
def load_npy(folder_path):
    real_imgs = []
    fake_imgs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy') and 'real' in filename:
            file_path = os.path.join(folder_path, filename)
            real_imgs.append(np.load(file_path))
        elif filename.endswith('.npy') and 'fake' in filename:
            file_path = os.path.join(folder_path, filename)
            fake_imgs.append(np.load(file_path))
    return np.asarray(real_imgs), np.asarray(fake_imgs)
    
def generate_gaussian_array(mean, cov, grid_size, num_points):
    """
    Genera un array 2D di dimensione grid_size x grid_size con una distribuzione gaussiana.

    Parameters:
        mean (list): La media della distribuzione gaussiana (es. [0, 0]).
        cov (list): La matrice di covarianza della distribuzione (es. [[64, 0], [0, 64]]).
        grid_size (int): La dimensione dell'array quadrato risultante (es. 256).
        num_points (int): Il numero di punti da generare.

    Returns:
        np.ndarray: Un array 2D di dimensione grid_size x grid_size con i valori della distribuzione gaussiana.
    """
    # Genera i punti della distribuzione gaussiana 2D
    x, y = np.random.multivariate_normal(mean, cov, num_points).T
    x, y = np.int32(np.floor(x)), np.int32(np.floor(y))

    # Crea una griglia vuota per contare il numero di punti in ciascuna cella
    gaussian_array = np.zeros((grid_size, grid_size))

    for value1, value2 in zip(x,y):
        gaussian_array[value1, value2] +=1

    return gaussian_array

data_dir='/leonardo/home/userexternal/lfontana/ALL_ROT_npy_version/1024x1024/'

'''
PATH = "/leonardo_work/INA24_C3B13/PGWGan_results/synthetic_data_z_1.00/"
real_imgs, fake_imgs = load_npy(PATH)


# Parametri
mean = [128, 128]
cov = [[500, 0], [0, 500]]
grid_size = 256
num_points = 10000

real_list = [generate_gaussian_array(mean, cov, grid_size, num_points) for i in range(10)]
real_imgs = np.expand_dims(np.stack(arrays=real_list, axis=0), axis=-1)

fake_list = [generate_gaussian_array(mean, cov, grid_size, num_points) for i in range(10)]
fake_imgs = np.expand_dims(np.stack(arrays=fake_list, axis=0), axis=-1)
'''
def load_meta_data(redshift):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()#.sort_values(by=['mass', 'rot']).reset_index(drop=True)
    print(f"Data Shape: {meta_data.shape}")

    # Showing what all is in my data
    # get_unique(meta_data)
    
    return meta_data

def split_into_subsets(lst, subset_size=10):
    subsets = []
    lst_copy = lst[:]
    while len(lst_copy) >= subset_size:
        # Seleziona randomicamente un sottoinsieme di 'subset_size' elementi
        subset = random.sample(lst_copy, subset_size)
        subsets.append(subset)
        
        # Rimuovi gli elementi selezionati dalla lista originale
        lst_copy = [item for item in lst_copy if not any(np.array_equal(item, s) for s in subset)]
    
    # Aggiungi eventuali elementi rimanenti in un ultimo sottoinsieme (se presenti)
    if lst_copy:
        subsets.append(lst_copy)
    
    return subsets

meta_data = load_meta_data(0.03)
images = []

for idx, data_point in meta_data.iterrows():
    image_arr = np.load(data_dir + str(data_point['id']) + '.npy').astype('float32')
    image_arr = (image_arr - np.mean(image_arr)) / (image_arr + np.mean(image_arr))
    image_arr = image_arr.reshape((1024, 1024, 1)).astype('float32')
    image_arr = tf.image.resize(image_arr, (256, 256)).numpy()
    
    images.append(image_arr)

subsets = split_into_subsets(images)
fid_list = []

for i in range(20):
    a, b = np.random.randint(0, len(subsets) - 1), np.random.randint(0, len(subsets) - 1)
    real_imgs, fake_imgs = subsets[a], subsets[b]
    
    real_imgs, fake_imgs = np.repeat(real_imgs, 3, axis=3), np.repeat(fake_imgs, 3, axis=3)
    real_imgs, fake_imgs = scale_images(real_imgs, (299, 299, 3)), scale_images(fake_imgs, (299, 299, 3))
    
    print(real_imgs.dtype)
    print(fake_imgs.shape)
        
    # fid between images1 and images1
    fid_0 = calculate_fid(real_imgs, real_imgs)
    print('FID (same): %.3f' % fid_0)
    # fid between images1 and images2
    fid = calculate_fid(real_imgs, fake_imgs)
    print('FID (different): %.3f' % fid)
    fid_list.append(fid)
    
fid_array = np.asarray(fid_list)
print("Mean Fid after 20 test: %.3f" % np.mean(fid_array))
print("Associated error: %.3f" % (np.sqrt(np.var(fid_array))/20.0))


def prepare_real_images(meta_data, num_samples=15):
    real_images = []

    ## Apply normalization and reshape into 256x256 images
    for idx, data_point in meta_data.iterrows():
        image_arr = np.load(data_dir + str(data_point['id']) + '.npy').astype('float32')
        image_arr = (image_arr - np.mean(image_arr)) / (image_arr + np.mean(image_arr))
        image_arr = image_arr.reshape((1024, 1024, 1)).astype('float32')
        image_arr = tf.image.resize(image_arr, (256, 256)).numpy()
        
        real_images.append(image_arr)

    ## Obtain a 15 images, random choosen, subset
    subset = [random.choice(real_images) for i in range(num_samples)]

    ## Prepare the sampled images for the application of the InceptionV3
    real_set = np.repeat(subset, 3, axis=3)
    real_set = scale_images(real_imgs, (299, 299, 3))

    return real_set

def prepare_fake_images(synthetic_set, num_samples=15):
    ## Selecting num_samples images according to the shape of the data
    synth_imgs = synthetic_set[:num_samples, :, :, 0]
    synth_imgs = np.repeat(synth_imgs, 3, axis=3)
    synth_imgs = scale_images(synth_imgs, (299, 299, 3))

    return synth_imgs