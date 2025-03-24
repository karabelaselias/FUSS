import os, csv
import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from utils.geometry_util import laplacian_decomposition, get_operators
from utils.shape_util import read_shape, compute_geodesic_distmat, write_off


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--n_eig', type=int, default=300, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')
    parser.add_argument('--no_normalize', action='store_true', help='no normalization of face area.')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    n_eig = args.n_eig
    no_eig = args.no_eig
    no_normalize = args.no_normalize
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    if not no_eig:
        spectral_dir = os.path.join(data_root, 'diffusion')
        os.makedirs(spectral_dir, exist_ok=True)

    # read .off files
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    assert len(off_files) != 0

    # create a seperate folder with the scaled meshes
    scaled_off_path = os.path.join(data_root, 'off_scaled')
    os.makedirs(scaled_off_path, exist_ok=True)
    
    # read mesh info file
    header = ['file_name', 'mean_x', 'mean_y', 'mean_z', 'face_area']
    with open(os.path.join(data_root, 'mesh_info.csv'), 'w+', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for off_file in tqdm(off_files):
            verts, faces = read_shape(off_file)
            filename = os.path.basename(off_file)

            if not no_normalize:

                # calculate scaled mesh
                verts_mean = np.mean(verts, axis=-2, keepdims=True)
                verts -= verts_mean
                
                coords = verts[faces]
                vec_A = coords[:, 1, :] - coords[:, 0, :]
                vec_B = coords[:, 2, :] - coords[:, 0, :]
                face_areas = np.linalg.norm(np.cross(vec_A, vec_B, axis=-1), axis=1) * 0.5
                total_area = np.sum(face_areas)
                scale = np.sqrt(total_area)
                verts = verts / scale
                
                # save new verts and faces
                scaled_file = os.path.join(scaled_off_path, filename)
                write_off(scaled_file, verts, faces)

                # write mesh info
                data = [filename[:-4], verts_mean[0,0], verts_mean[0,1], verts_mean[0,2], scale]
                writer.writerow(data)

            if not no_eig:
                # recompute laplacian decomposition
                get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                              k=n_eig, cache_dir=spectral_dir)
