from data_utils import CustomDataGen, load_meta_data
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_image_grid(data_gen, num_images=9):
    """
    Plot a grid of images from the data generator with a shared color scale.

    Parameters:
        data_gen (CustomDataGen): The data generator instance.
        num_images (int): Number of images to display in the grid.
    """
    plt.figure(figsize=(10, 10))
    X, y = data_gen.__getitem__(0)  # Get the first batch

    # Determina la dimensione della griglia
    grid_size = int(np.sqrt(num_images))

    # Crea i subplot e plottali con la stessa colormap e scala
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Spazio extra per evitare sovrapposizioni tra immagini e titoli
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    for i in range(num_images):
        ax = axes[i // grid_size, i % grid_size]
        img = ax.imshow(X[i], cmap='inferno')  # Imposta vmin e vmax comuni
        ax.set_title(f"Mass: {y[i]:.2f}", fontsize=10, color='white')  # Titolo in bianco
        ax.axis('off')

        # Aggiungi una colorbar accanto a ciascun subplot
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        # Forza la notazione scientifica sulla colorbar
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-5, -4))
        cbar.ax.yaxis.set_major_formatter(formatter)

        # Imposta il colore delle etichette della colorbar su bianco
        cbar.ax.yaxis.set_tick_params(color='white')  # Colore dei tick
        cbar.outline.set_edgecolor('white')  # Colore del bordo della colorbar
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')  # Colore delle etichette
        cbar.ax.yaxis.get_offset_text().set_color('white')  # Colore dell'esponente in notazione scientifica

        # Imposta il colore del bordo (spine) dei subplot su bianco
        '''
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        '''

    # Salva il grafico senza sfondo
    plt.savefig('example_grid.png', bbox_inches='tight', transparent=True)
    plt.close()

z = 0.39
meta_data = load_meta_data(z)

batch_size = 32
target_size = (256, 256)
X_col = 'id'
y_col = 'mass'
    
# Create data generator
data_gen = CustomDataGen(meta_data, X_col, y_col, batch_size, target_size)

# Plot the image grid
plot_image_grid(data_gen)
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model_utils import*
from tensorflow.keras.models import Model

REDSHIFT = 0.39
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 16]
FILTERS = [512, 256, 128, 64, 32, 16, 8]
REGRESSOR_FILTERS = [50, 50, 50, 50, 20, 10, 10]
REGRESSOR_FILTERS_2 = [50, 50, 50, 20, 10, 10, 10]
GP_WEIGHT = 10,
DRIFT_WEIGHT=0.001


class PGAN(Model):
    def __init__(self, latent_dim, d_steps, gp_weight = GP_WEIGHT, drift_weight = DRIFT_WEIGHT):
        super(PGAN, self).__init__()
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_depth = 0
        self.discriminator = self.init_discriminator()
        self.discriminator_wt_fade = None
        self.generator = self.init_generator()
        self.regressor = self.init_regressor()

        self.generator_wt_fade = None

    def call(self, inputs):
        return

    def init_discriminator(self):
        img_input = tf.keras.layers.Input(shape = (4,4,1))
        # print(f" \n \n {img_input.shape} \n \n")
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # fromGrayScale
        x = WeightScalingConv(img_input, filters = FILTERS[0], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x) # 4 x 4 x 513

        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4,4)) # 1 x 1 x 512

        x = tf.keras.layers.Flatten()(x)
        
        x = WeightScalingDense(x, filters=1, gain=1.)
        # print("\n \n Dense Layer Output Type:", type(x))
        d_model = Model(img_input, x, name='discriminator')

        return d_model

    # Fade in upper resolution block
    def fade_in_discriminator(self):

        input_shape = list(self.discriminator.input.shape) 
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape = input_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        #    Reuse the existing “FromGrayScale” block defined as “x1" -- SKIP CONNECTION (ALREADY STABILIZED -> 1-alpha)
        x1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(img_input) # 4 x 4 x 1
        x1 = self.discriminator.layers[1](x1) # Conv2D FromGrayScale # 4 x 4 x 512
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer # 4 x 4 x 512
        x1 = self.discriminator.layers[3](x1) # Bias # 4 x 4 x 512
        x1 = self.discriminator.layers[4](x1) # LeakyReLU # 4 x 4 x 512

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        # symmetric
        x2 = WeightScalingConv(img_input, filters = FILTERS[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256

        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256
        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 512

        x2 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(x2) # 4 x 4 x 512
        # print(f"\n \n {x2.shape} \n \n")

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model(img_input, x2, name='discriminator')

        # 5. Add existing discriminator layers. 
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model(img_input, x, name='discriminator')

    # Change to stabilized(c. state) discriminator 
    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        
    def init_regressor(self):
        
        img_input = tf.keras.layers.Input(shape = (4, 4, 1))
        img_input = tf.keras.ops.cast(img_input, tf.float32)
                
        #  [(I - F +2 *P) / S] +1 = 4 x 4 x 50

        x = RegressorConv(img_input, REGRESSOR_FILTERS[0], kernel_size = 1, pooling=None, activate='LeakyReLU', strides=(1,1))
        
        
        # print(x.shape) # 4 x 4 x 50
        x = RegressorConv(x, REGRESSOR_FILTERS[0], kernel_size = 3, pooling='avg', activate='LeakyReLU', strides=(1,1)) 
        # print(x.shape) # should be 1 x 1 x 50
        x = tf.keras.layers.Flatten()(x) # 50
        x = tf.keras.layers.Dense(units = 16)(x) # 16
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        
        x = tf.keras.layers.Dense(units = 1)(x) # 1

        c_model = Model(img_input, x, name='regressor')

        return c_model

    def fade_in_regressor(self):

        input_shape = list(self.regressor.input.shape)
        
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape = input_shape)
        img_input = tf.keras.ops.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        x1 = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2))(img_input) 
        x1 = self.regressor.layers[1](x1) # Conv2D 
        x1 = self.regressor.layers[2](x1) # BatchNormalization 
        x1 = self.regressor.layers[3](x1) # LeakyReLU 

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        
        if self.n_depth!=5:
            x2 = RegressorConv(img_input, REGRESSOR_FILTERS_2[self.n_depth], kernel_size = 1, pooling=None, activate='LeakyReLU', strides=(1,1))

            x2 = RegressorConv(x2, REGRESSOR_FILTERS[self.n_depth], kernel_size = 3, pooling='max', activate='LeakyReLU', strides=(1,1))
            
        else:
            x2 = RegressorConv(img_input, REGRESSOR_FILTERS[self.n_depth], kernel_size = 3, pooling='max', activate='LeakyReLU', strides=(1,1))

        
        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(4, len(self.regressor.layers)):
            x2 = self.regressor.layers[i](x2)
        self.regressor_stabilize = Model(img_input, x2, name='regressor')

        # 5. Add existing discriminator layers. 
        for i in range(4, len(self.regressor.layers)):
            x = self.regressor.layers[i](x)
        self.regressor = Model(img_input, x, name='regressor')

    # Change to stabilized(c. state) discriminator 
    def stabilize_regressor(self):
        self.regressor = self.regressor_stabilize

    def init_generator(self):
        noise = tf.keras.layers.Input(shape=(self.latent_dim,)) # None, 512
        mass = tf.keras.layers.Input(shape=(1,))
        
        merge = tf.keras.layers.Concatenate()([noise, mass]) #L x (3)
                
        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
 
        x = WeightScalingDense(merge, filters=4*4*FILTERS[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=False) 
        
        x = tf.keras.layers.Reshape((4, 4, FILTERS[0]))(x)

        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=False)
        
        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)

        # Gain should be 1 as its the last layer 
        x = WeightScalingConv(x, filters=1, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # change to tanh and understand gain 1 if training unstable

        g_model = Model([noise,mass], x, name='generator')

        return g_model

    # Fade in upper resolution block
    def fade_in_generator(self):

        # 1. Get the node above the “toGrayScale” block 
        block_end = self.generator.layers[-5].output
        
        # 2. Upsample block_end       
        block_end = tf.keras.layers.UpSampling2D((2,2))(block_end) # 8 x 8 x 512

        # 3. Reuse the existing “toGrayScale” block defined as“x1”. --- SKIP CONNECTION (ALREADY STABILIZED)
        x1 = self.generator.layers[-4](block_end) # Conv2d
        x1 = self.generator.layers[-3](x1) # WeightScalingLayer
        x1 = self.generator.layers[-2](x1) # Bias
        x1 = self.generator.layers[-1](x1) # tanh

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".
        x2 = WeightScalingConv(block_end, filters = FILTERS[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512 
        
        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512 
        
        # "toGrayScale"
        x2 = WeightScalingConv(x2, filters=1, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # 

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

    # Change to stabilized(c. state) generator 
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

    def compile(self, d_optimizer, g_optimizer, r_optimizer):
        super(PGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.r_optimizer = r_optimizer


xgan = PGAN(latent_dim = 512, d_steps = 5)

for n_depth in range(1,7):
  xgan.n_depth = n_depth
  xgan.fade_in_generator()
  xgan.fade_in_discriminator()
  xgan.fade_in_regressor()

  xgan.stabilize_generator()
  xgan.stabilize_discriminator()
  xgan.stabilize_regressor()


ckpt_path = "/leonardo_work/INA24_C3B13/PGWGan_results/VAC-PGGAN_ckpts_z_0.39/pgan_6_stabilize/pgan_00160.weights.h5"
xgan.load_weights(ckpt_path)

# tf.keras.utils.plot_model(xgan.generator, to_file='generator_6.png', show_shapes=True)
# tf.keras.utils.plot_model(xgan.discriminator, to_file='discriminator_6.png', show_shapes=True)
# tf.keras.utils.plot_model(xgan.regressor, to_file='regressor_6.png', show_shapes=True)


def load_meta_data(redshift):
    meta_data = pd.read_csv("mainframe.csv")
    meta_data=meta_data[meta_data['redshift']==redshift]

    meta_data = meta_data[['id','redshift', 'mass', 'simulation', 'snap', 
                           'ax', 'rot']].drop_duplicates()#.sort_values(by=['mass', 'rot']).reset_index(drop=True)
    print(f"Data Shape: {meta_data.shape}")

    # Showing what all is in my data
    # get_unique(meta_data)
    
    return meta_data


meta_data = load_meta_data(REDSHIFT)
masses = [14.57, 14.89, 14.86, 14.84, 14.99, 14.51, 14.61, 14.22, 14.70, 14.77]

def generate_image(mass, pgan, num_imgs, noise_dim=512, typeof='fake', real_df=meta_data, save_dir=f'synthetic_data_z_{REDSHIFT}/', data_dir='/leonardo/home/userexternal/lfontana/ALL_ROT_npy_version/1024x1024/'):
    # Verifica che mass sia una lista con num_imgs elementi
    if not isinstance(mass, list) or len(mass) != num_imgs:
        raise ValueError("La variabile 'mass' deve essere una lista contenente num_imgs elementi.")
    
    # Creare la directory di salvataggio se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if typeof == 'fake':
        random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
        mass = np.array(mass).reshape(num_imgs, 1)

        generated_imgs = pgan.generator([random_latent_vectors, tf.convert_to_tensor(mass, dtype=tf.float32)])

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
            image_arr = image_arr.reshape((1024, 1024, 1)).astype('float32')
            image_arr = tf.image.resize(image_arr, (256, 256)).numpy()

            # Salva l'immagine come file png
            image_path = os.path.join(save_dir, f'real_image_{i}_mass_{data_point["mass"]}')
            np.save(image_path, image_arr)
        print(f'Saved {num_imgs} real images with varying masses in {save_dir}')

generate_image(masses, xgan, 10, typeof='real')
'''
'''
import numpy as np
from matplotlib import pyplot as plt

img_name = "fake_image_2_mass_14.74"
mean = 3.35e-06

array = np.resize(np.load(f"/leonardo/home/userexternal/lfontana/GAN/Paper_code/synthetic_data_z_0.03/{img_name}.npy"), (256,256))
inverted = - (array + 1) * mean
inverted = inverted / (array - 1)
plt.imsave(f"synthetic_data_z_0.03/{img_name}.png", inverted, cmap='inferno')
'''
'''
from test_utils import plot_loss

plot_loss("/leonardo/home/userexternal/lfontana/GAN/Paper_code/VAC-PGGAN_Loss_z_1.00/")
'''