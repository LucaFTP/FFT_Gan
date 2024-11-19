from model import *
from train import *
from gan_utils import *
from data_utils import *
from gan_utils import *
from model_utils import *
# import wandb

# wandb.init()

REDSHIFT = 0.03
D_STEPS = 5
EPOCHS = 2
NOISE_DIM = 512
NUM_IMGS_GENERATE = 9
STEPS_PER_EPOCH = 7
START_SIZE = 4
END_SIZE = 256
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 16]


meta_data = load_meta_data(REDSHIFT, print_opt=True)


pgan = PGAN(latent_dim = NOISE_DIM, d_steps = D_STEPS)
cbk = GANMonitor(num_img = NUM_IMGS_GENERATE, latent_dim = NOISE_DIM, redshift=REDSHIFT)
cbk.set_steps(steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS) # 110, 6
cbk.set_prefix(prefix='0_init')

# Cluster - WandB sweep
# pgan = train(wandb.config.G_LR, wandb.config.D_LR, wandb.config.R_LR, EPOCHS, D_STEPS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, aug_meta_data)

# Local
pgan = train(0.001, 0.001, 0.001, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, meta_data)

# Save the values of the FID score at the end of training
np.save("fid_scores", cbk.fid_scores)

tstr = compute_tstr(meta_data= meta_data, model=pgan, d_steps=D_STEPS, NOISE_DIM=NOISE_DIM)
print(f"TSTR: {tstr}")

# wandb.log({"tstr": tstr})