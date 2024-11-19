import os
import pandas as pd

root_dir = "/u/13465/GAN/ALL_LOG_ROT/npy_version"
images = [f for f in os.listdir(root_dir) if f.endswith('.npy')]

colNames = {'id': [], 'redshift': [], 'mass': [],
            'simulation': [], 'snap': [], 'ax': [], 'rot': []}
mainframe = pd.DataFrame(data = colNames)

for img in images:
    img_name = os.path.join(root_dir, img)
    id = img_name.split('.npy')[0].split(root_dir + '/')[1]
    z = float(id.split('reds=')[1].split('_ax')[0])
    mass = float(id.split('mass=')[1].split('_reds')[0])
    sim = id.split('sim=')[1].split('_snap')[0]
    snap = id.split('snap=')[1].split('_mass')[0]
    ax = id.split('ax=')[1].split('_rot')[0]
    rot = id.split('rot=')[1]

    vals = {'id': [id], 'redshift': [z], 'mass': [mass],
            'simulation': [sim], 'snap': [snap], 'ax': [ax], 'rot': [rot]}
    tempdf = pd.DataFrame(data = vals)
    frames = [mainframe, tempdf]
    mainframe = pd.concat(frames)

mainframe.to_csv('mainframe.csv')

