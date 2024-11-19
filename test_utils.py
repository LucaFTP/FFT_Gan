import re
import matplotlib.pyplot as plt
import imageio
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.linalg import sqrtm
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

data_dir = "/leonardo/home/userexternal/lfontana/ALL_ROT_npy_version/1024x1024/"
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

def generate_image(mass, pgan, num_imgs, noise_dim, typeof='fake', real_df=meta_data, save_dir='./generated_images/'):
    # Verifica che mass sia una lista con num_imgs elementi
    if not isinstance(mass, list) or len(mass) != num_imgs:
        raise ValueError("La variabile 'mass' deve essere una lista contenente num_imgs elementi.")
    
    # Creare la directory di salvataggio se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if typeof == 'fake':
        random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
        mass = np.array(mass).reshape(num_imgs, 1)

        generated_imgs = pgan.generator([random_latent_vectors, mass])

        for i in range(num_imgs):
            # Salva l'immagine generata come file numpy
            image_path = os.path.join(save_dir, f'fake_image_{i}_mass_{mass[i][0]}.npy')
            np.save(image_path, generated_imgs[i].numpy())
        print(f'Saved {num_imgs} fake images with varying masses in {save_dir}')

    elif typeof == 'real':
        mass_df = real_df[real_df['mass'].isin(mass)]
        selected_data = []

        for m in mass:
            data_point = mass_df[mass_df['mass'] == m].iloc[np.random.randint(low=0, high=mass_df[mass_df['mass'] == m].shape[0])]
            selected_data.append(data_point)

        for i, data_point in enumerate(selected_data):
            image_arr = np.load(data_dir + data_point['id'] + '.npy').astype('float32')
            image_arr = (image_arr - np.mean(image_arr)) / (image_arr + np.mean(image_arr))
            image_arr = image_arr.reshape((256, 256, 1)).astype('float32')
            image_arr = tf.image.resize(image_arr, (256, 256)).numpy()

            # Salva l'immagine come file numpy
            image_path = os.path.join(save_dir, f'real_image_{i}_mass_{data_point["mass"]}.npy')
            np.save(image_path, image_arr)
        print(f'Saved {num_imgs} real images with varying masses in {save_dir}')
        
def generate_image_multitask(mass, redshift, pgan, CustomDataGen, num_imgs, noise_dim, typeof = 'fake', plot_wandb=False, real_df = meta_data):
                                            
    if typeof == 'fake':
        random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
        mass = np.ones([num_imgs, 1]) * mass
        redshift = np.ones([num_imgs,1]) * redshift
        generated_imgs = pgan.generator([random_latent_vectors, mass, redshift])
        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i, :, :, 1], cmap='afmhot')
            ax.axis('off')
        
        fig.suptitle(f'Fake Images with mass: {mass[0][0]}, redshift: {redshift[0][0]}')
        plt.show()
        '''
        if plot_wandb:
            log_dict = {
            "Generated Images while evaluating (345 GHz)": [wandb.Image(generated_imgs[i, :, :, 1], caption='a = ' + str(mass[i][0])+', redshift = ' + str(redshift[i][0])) for i in range(num_imgs)]}
            
            wandb.log(log_dict)
        '''
    if typeof == 'real':
        filtered_df = real_df[real_df['mass']==mass]
        filtered_df = filtered_df[filtered_df['redshift']==redshift] # int(2**(redshift * np.log2(160)))
        filtered_ds = CustomDataGen(filtered_df, X_col='id', y_col=['mass','redshift'], rot_col = False, batch_size = 10, target_size=(128,128), 
                              freqs = ['fixed'], blur = 0, shuffle=True)

        print('Redshift z:', redshift)


        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(filtered_ds[np.random.randint(0, len(filtered_ds)-1)][0][i,:,:,1], cmap='afmhot') # first batch, X, image
            ax.axis('off')

        fig.suptitle(f'Real Images with mass: {mass}, Redshift: {redshift}')
        plt.show()
        
def compare_images(real_df, mass, pgan, noise_dim):

    
    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    ax = ax.flatten()
    
    # Real Image
    mass_df = real_df[real_df['mass']==mass]
    data_point = mass_df.iloc[np.random.randint(low = 0, high = mass_df.shape[0])]
    path = data_point['id']
    mass = data_point['mass']
    image_arr = np.load(data_dir + path +'.npy').astype('float32')
    image_arr = (image_arr - np.mean(image_arr))/(image_arr + np.mean(image_arr))
    image_arr = image_arr.reshape((160, 160, 1)).astype('float32')  
    image_arr = tf.image.resize(image_arr,(128, 128)).numpy() 
    ax[0].imshow(image_arr, cmap='afmhot')
    ax[0].set_title(f'Real Image - mass: {mass}')
    ax[0].axis('off')
    
    # Fake Image
    random_latent_vectors = tf.random.normal(shape=[1, noise_dim])
    mass = np.ones([1, 1]) * mass
    generated_imgs = pgan.generator([random_latent_vectors, mass])
    ax[1].imshow(generated_imgs[0, :, :, 0], cmap='afmhot')
    ax[1].set_title(f'Fake Image - mass: {mass[0][0]}')
    ax[1].axis('off')
    
    
    plt.show()
        
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
    real_set = scale_images(real_set, (299, 299, 3))

    return real_set

def prepare_fake_images(synthetic_set, num_samples=15):
    ## Selecting num_samples images according to the shape of the data
    synth_imgs = synthetic_set[:num_samples, :, :, :]
    synth_imgs = np.repeat(synth_imgs, 3, axis=3)
    synth_imgs = scale_images(synth_imgs, (299, 299, 3))

    return synth_imgs