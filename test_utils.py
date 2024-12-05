import re
import matplotlib.pyplot as plt
import imageio
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.linalg import sqrtm
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

data_dir = "/leonardo_scratch/fast/INA24_C3B13/ALL_ROT_npy_version/1024x1024/"
meta_data = pd.DataFrame()

def create_gif(path):
    image_folder = os.fsencode(path)

    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.jpeg', '.png', '.gif') ):
            filenames.append(filename)

    filenames.sort() # this iteration technique has no built in order, so sort the frames
    images = list(map(lambda filename: imageio.v2.imread(f'{path}/'+filename), filenames))

    return filenames,images

def load_epoch_weights(PGAN, latent_dim, ckpt_epoch_path):
    xgan = PGAN(latent_dim = latent_dim)


    for n_depth in range(1,7):
        xgan.n_depth = n_depth
        xgan.fade_in_generator()
        xgan.fade_in_discriminator()
        xgan.fade_in_regressor()

        xgan.stabilize_generator()
        xgan.stabilize_discriminator()
        xgan.stabilize_regressor()


    xgan.load_weights(ckpt_epoch_path)

    return xgan
        
def plot_loss(loss_path):
    START_SIZE = 4
    num_files = (len(os.listdir(loss_path)) + 1)
    fig, ax = plt.subplots(num_files//2,2, figsize=(15,25))
    ax = ax.flatten()
    i = 0
    s={}
    color = ['b','g','r','y']
    for file in os.listdir(loss_path):
        name = re.split('_|\.',file)[1]
        iteration = re.split('_|\.',file)[-2]
        if name in ['init', 'stabilize']:
            s[name + iteration] = np.load(loss_path+'/'+file,allow_pickle=True)
    s = sorted(s.items())
    for j in range(len(s)):
        ax[i].plot(s[j][1].item()['d_loss'], '.-')
        ax[i].plot(s[j][1].item()['g_loss'], '.-')

        ax[i+1].plot(s[j][1].item()['r_loss'], '.-')

        try:
            IMG_SIZE = 2**(2+j)
            ax[i].set_title(f"Image Size: {IMG_SIZE} x {IMG_SIZE}")
            ax[i+1].set_title(f"Image Size: {IMG_SIZE} x {IMG_SIZE}")
        except:
            ax[i].set_title(f"Image Size: {START_SIZE} x {START_SIZE}")
            ax[i+1].set_title(f"Image Size: {START_SIZE} x {START_SIZE}")
        ax[i].legend(['Discriminator Loss', 'Generator Loss'])
        ax[i+1].legend(['Generated Mass Loss', 'Real Mass Loss'])

        i = i + 2
        
    # Salva la figura
    output_path = loss_path + "image.png"
    plt.savefig(output_path, bbox_inches='tight')

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

def prepare_real_images(meta_data, num_samples=15):
    real_images = []

    meta_data = meta_data[meta_data['rot'] == 0]
    ## Apply normalization and reshape into 64x64 images
    for idx, data_point in meta_data.iterrows():
        image_arr = np.load(data_dir + str(data_point['id']) + '.npy').astype('float32')
        image_arr = (image_arr - np.mean(image_arr)) / (image_arr + np.mean(image_arr))
        image_arr = image_arr.reshape((512, 512, 1)).astype('float32')
        image_arr = tf.image.resize(image_arr, (64, 64)).numpy()
        
        real_images.append(image_arr)

    ## Obtain a 15 images, random choosen, subset
    subset = real_images#[random.choice(real_images) for i in range(num_samples)]

    ## Prepare the sampled images for the application of the InceptionV3
    real_set = np.repeat(subset, 3, axis=3)
    real_set = scale_images(real_set, (299, 299, 3))

    return real_set

def prepare_fake_images(synthetic_set, num_samples=15):
    ## Selecting num_samples images according to the shape of the data
    synth_imgs = synthetic_set[:num_samples, :, :, :]
    synth_imgs = np.repeat(synth_imgs, 3, axis=3)
    synth_imgs = scale_images(synth_imgs, (299, 299, 3))

    return synth_imgs