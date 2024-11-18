import os
import numpy as np
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def create_folders(redshift):
    
    version='_z_'+redshift
    CKPT_OUTPUT_PATH = 'FFT_GAN_ckpts'+version
    IMG_OUTPUT_PATH = 'FFT_GAN_Images'+version
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
        os.mkdir(LOSS_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(DATASET_OUTPUT_PATH)
    except FileExistsError:
        pass
    
    return CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH, DATASET_OUTPUT_PATH

# Definizione del dataset personalizzato
class NpyDataset(Dataset):
    def __init__(self, root_dir, img_size, redshift, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.redshift = "reds="+str(redshift)
        self.npy_files = [f for f in os.listdir(root_dir) if f.endswith('.npy') and self.redshift in f]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
      while True:
        img_name = os.path.join(self.root_dir, self.npy_files[idx])
        image = block_mean(np.load(img_name), 1024 // self.img_size)

        # Applica le trasformazioni, se definite
        if self.transform:
            image = self.transform(image)
            
        return image
      
# Funzione per caricare il dataset e creare il DataLoader
def create_dataloader(root_dir, batch_size, img_size, shuffle=True, num_workers=0):
    # Definizione delle trasformazioni
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
            ])

    dataset = NpyDataset(root_dir=root_dir, img_size=img_size, transform=transform)            
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader
    
def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res