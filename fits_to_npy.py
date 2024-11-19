import os
import numpy as np
from astropy.io import fits

root_dir = "/leonardo_work/INA24_C3B13/ALL_ROT/"
output_dir = "/leonardo/home/userexternal/lfontana/ALL_ROT_npy_version/1024x1024/"

fits_files = [f for f in os.listdir(root_dir) if f.endswith('.fits')]
counter = 0
not_used = []
for f in fits_files:
    img_name = os.path.join(root_dir, f)
    with fits.open(img_name) as hdul:
            image = hdul[0].data
            image = image.astype('float32')#[256:768, 256:768]

            useful_part = img_name.split("ALL_ROT/")[1].split('_log_')[0]
            if len(useful_part) == 29:
                  sim  = '_sim=' + useful_part[4:7]
                  snap = 'snap=' + useful_part[7:11]
                  mass = 'mass=' + useful_part[11:17]
                  reds = 'reds=' + useful_part[17:21]
                  ax = '_ax=' + useful_part[-1]

            elif len(useful_part) == 30:
                  sim  = '_sim=' + useful_part[4:8]
                  snap = 'snap=' + useful_part[8:12]
                  mass = 'mass=' + useful_part[12:18]
                  reds = 'reds=' + useful_part[18:22]
                  ax = '_ax=' + useful_part[-1]
            
            else: not_used.append(useful_part)

            rot = '_rot=' + img_name.split('_rot')[1].split('_')[0]
            '''
            print(img_name.split(root_dir + "/")[1])
            print(f"{sim} \n {snap} \n {mass} \n {reds} \n {ax} \n {rot}")
            print("tSZ" + sim + snap + mass + reds + ax + rot)

            counter += 1
            if counter == 10:
                print("not used:") 
                print(not_used)
                break
            '''
            if image.max() != image.min():
                new_name = "tSZ" + sim + snap + mass + reds + ax + rot
                out_path = output_dir + new_name
                np.save(out_path, image)

print(f"All the images have been saved in {output_dir} as .npy files")

print(os.path.getsize(out_path + '.npy'), 'bytes')

im_r_pil = np.load(out_path + '.npy') 
print(f"Dimension check: {im_r_pil.size}")

