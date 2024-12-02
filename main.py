import argparse
from matplotlib import pyplot

from model import *
from train import *
from main_utils import *
from gan_utils import *
from data_utils import *
from gan_utils import *
from model_utils import *
# import wandb

# wandb.init()

# Parser creation
parser = argparse.ArgumentParser(
    description="Imposta i parametri per l'esecuzione del programma."
)
# Parser parameter definition
parser.add_argument("-z", "--redshift", type=float, required=True,
                    help="Redshift value (obbligatorio).")
parser.add_argument("--d_steps", type=int, default=5,
                    help="Discriminator steps / Generator steps (default: 5).")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of epochs (default: 50).")
parser.add_argument("--noise_dim", type=int, default=512,
                    help="Latent noise vector dimension (default: 512).")
parser.add_argument("--num_imgs_generate", type=int, default=9,
                    help="Number of images to generate (default: 9).")
parser.add_argument("--steps_per_epoch", type=int, default=7,
                    help="Steps per epoch (default: 7).")
parser.add_argument("--end_size", type=int, default=64,
                    help="Target size of the images in the final step (default: 64).")
parser.add_argument("--batch_size", nargs="+", type=int, default=[32, 32, 16, 16, 16, 8, 8, 4],
                    help="Batch size in every level (default: [32, 32, 16, 16, 16, 8, 8, 4]).")
# Arguments parsing
args = parser.parse_args()

# Validate the END_SIZE input value
validate_end_size(args.end_size)

# Print of parser parameters
print(f"REDSHIFT: {args.redshift}")
print(f"D_STEPS: {args.d_steps}")
print(f"EPOCHS: {args.epochs}")
print(f"NOISE_DIM: {args.noise_dim}")
print(f"NUM_IMGS_GENERATE: {args.num_imgs_generate}")
print(f"STEPS_PER_EPOCH: {args.steps_per_epoch}")
print(f"END_SIZE: {args.end_size}")
print(f"BATCH_SIZE: {args.batch_size}")

REDSHIFT = args.redshift
D_STEPS = args.d_steps
EPOCHS = args.epochs
NOISE_DIM = args.noise_dim
NUM_IMGS_GENERATE = args.num_imgs_generate
STEPS_PER_EPOCH = args.steps_per_epoch
END_SIZE = args.end_size
BATCH_SIZE = args.batch_size

CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, ARCH_OUTPUT_PATH, LOSS_OUTPUT_PATH, DATASET_OUTPUT_PATH = create_folders(redshift=REDSHIFT)

meta_data = load_meta_data(REDSHIFT, show=True)
print(f"Data Shape: {meta_data.shape}")

pgan = PGAN(latent_dim = NOISE_DIM, d_steps = D_STEPS)
cbk = GANMonitor(num_img = NUM_IMGS_GENERATE, latent_dim = NOISE_DIM, redshift=REDSHIFT,
                 checkpoint_dir=CKPT_OUTPUT_PATH, image_path=IMG_OUTPUT_PATH)
cbk.set_steps(steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS) # 110, 6
cbk.set_prefix(prefix='0_init')

# Cluster - WandB sweep
# pgan = train(wandb.config.G_LR, wandb.config.D_LR, wandb.config.R_LR, EPOCHS, D_STEPS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, aug_meta_data)

# Local
pgan = train(0.001, 0.001, 0.001, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, END_SIZE,
             cbk, pgan, meta_data, loss_out_path=LOSS_OUTPUT_PATH)

# Save the values of the FID score at the end of training
np.save("fid_scores", cbk.fid_scores)

tstr = compute_tstr(meta_data= meta_data, model=pgan, d_steps=D_STEPS, NOISE_DIM=NOISE_DIM, END_SIZE=END_SIZE)
print(f"TSTR: {tstr}")

# wandb.log({"tstr": tstr})