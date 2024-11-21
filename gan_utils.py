import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
# import wandb
from data_utils import *
from model_utils import *
from test_utils import prepare_fake_images, prepare_real_images, calculate_fid

    
def plot_models(pgan, ARCH_OUTPUT_PATH, typeof='init'):

    if typeof=='fade in':
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}_fade_in.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}_fade_in.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}_fade_in.png', show_shapes=True)
        
    elif typeof=='stabilize':
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}_stabilize.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}_stabilize.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}_stabilize.png', show_shapes=True)
        
    else:
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}.png', show_shapes=True)

# Saves generated images and updates alpha in WeightedSum layers
class GANMonitor(tf.keras.callbacks.Callback):
    
    def __init__(self, num_img, latent_dim, redshift, prefix='', checkpoint_dir = '', image_path = ''):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.redshift = redshift
        self.random_latent_vectors = tf.random.normal(shape=[num_img, self.latent_dim])
        self.mass = tf.convert_to_tensor(np.round(tf.random.uniform(
                        shape=[num_img, 1], minval=13,maxval=15),2))
        self.steps_per_epoch = 0
        self.epochs = 0
        self.steps = self.steps_per_epoch * self.epochs
        self.n_epoch = 0
        self.prefix = prefix
        self.checkpoint_dir = checkpoint_dir
        self.image_path = image_path
        self.absolute_epoch = 0
        self.fid_scores = []
        
  
    def set_prefix(self, prefix=''):
        self.prefix = prefix
        
    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs # 660

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch
        checkpoint_path = f"{self.checkpoint_dir}/pgan_{self.prefix}/pgan_{self.n_epoch:05d}.weights.h5"
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):

        prefix_number = int(self.prefix.split("_")[0])
        prefix_state = str(self.prefix.split("_")[1])

        # Plot epoch end generated images
        n_grid = int(np.sqrt(self.num_img))
        generated_imgs = self.model.generator([self.random_latent_vectors,self.mass])
        
        '''
        fig, axes = plt.subplots(n_grid,n_grid,figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i, :, :, 0], cmap='inferno') 
            img_mass = self.mass[i][0]
            ax.set_title(f'mass: {tf.get_static_value(img_mass):.02f}')
            ax.axis('off')
            plt.savefig(f'{IMG_OUTPUT_PATH}/plot_{self.prefix}_{epoch:05d}.jpeg')
        plt.suptitle('tSZ', fontsize = 15, fontweight = 'bold')
        plt.show()
        plt.close()
        '''
        if epoch % 30 == 0:
            plt.figure(figsize=(10, 10))
            for i in range(self.num_img):
                plt.subplot(n_grid, n_grid, i+1)
                plt.imshow(generated_imgs[i, :, :, 0], cmap='inferno')
                img_mass = self.mass[i][0]
                plt.title(f'mass: {tf.get_static_value(img_mass):.02f}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.image_path}/plot_{self.prefix}_{epoch:05d}.png')
            plt.close()

        ''' 
        image_size = (4*(2**prefix_number), 4*(2**prefix_number))

        if (prefix_state=='stabilize') or (prefix_state=='init'):

            log_dict = {
            "Generated Images while training (345 GHz)": [wandb.Image(generated_imgs[i, :, :, 1], 
                                            caption='a = ' + str(self.mass[i][0])) for i in range(self.num_img)],
            f'Discriminator Loss ({image_size[0]}x{image_size[0]})': logs['d_loss'],
            f'Generator Loss ({image_size[0]}x{image_size[0]})': logs['g_loss'],
            f'Spin Loss on real data ({image_size[0]}x{image_size[0]})': logs['r_loss'], 
            f'Epoch':self.n_epoch
            }
            wandb.log(log_dict) # was commit=False and was working, but only logged last value, working without it as well
            
        '''
        if (prefix_number == 4):
            ## Calculate the FID score after every epoch end between 15-images-sets
            meta_data = load_meta_data(self.redshift)
            real_images = prepare_real_images(meta_data=meta_data)
            synthetic_images = prepare_fake_images(generated_imgs)
            self.fid_scores.append(calculate_fid(real_images, synthetic_images))
            if (prefix_state=='stabilize') and ((epoch%5==0) or (epoch==self.epochs-1)):
                print('Saving weights...')
                self.model.save_weights(self.checkpoint_path)
                print('Successfuly saved weights.')
        '''
        if (prefix_number == 5) and (prefix_state=='stabilize') and ((epoch%5==0) or (epoch==self.epochs-1)):
            print('Saving weights...')
            self.model.save_weights(self.checkpoint_path)
            print('Successfuly saved weights.')
           
        if (prefix_number == 4) and (prefix_state=='stabilize') and (epoch==self.epochs-1):
            print('Saving weights...')
            self.model.save_weights(self.checkpoint_path)
            print('Successfuly saved weights.')
        
        if (prefix_number == 6) and (prefix_state=='stabilize') and (epoch==self.epochs-1):
            print('Saving model...')
            self.model.save(self.checkpoint_path)
            print('Successfuly saved model.')
        '''
            
    def on_batch_begin(self, batch, logs=None):
        
        # Update alpha in WeightedSum layers
        # alpha usually goes from 0 to 1 evenly over ALL the epochs for that depth.
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1) #1/219  to 1*110+109/220 for 2 epochs
        
        # print(f'!!! From GANMonitor: Steps: {self.steps}, Epoch: {self.n_epoch}, Steps per epoch: {self.steps_per_epoch}, Batch: {batch}, Alpha: {alpha}')
        
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
                
