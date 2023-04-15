import trimesh
from scipy.io import loadmat, savemat
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Draw 3D face', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', dest='path', type=str, default=False,help='The path of image folder')
    return parser.parse_args()
args = get_args()

img_folder = args.path

mesh_data = loadmat(os.path.join(img_folder, 'mean_face.mat'))
face = mesh_data['faces'].astype(np.uint64) - 1
vertices = mesh_data['vertices']
print(vertices.shape)

stitch = np.loadtxt('stitch_bfm.csv', delimiter=',')
print(stitch)
face = np.vstack((face, stitch))

mesh = trimesh.Trimesh(vertices=vertices, faces=face)
mesh.export(os.path.join(img_folder, 'mean_face_stitch.stl'))

face = face + 1
savemat('mean_face_stitch.mat', {'faces': face, 'vertices': vertices})
