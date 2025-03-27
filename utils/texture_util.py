import os
import numpy as np
import open3d as o3d

def write_obj_pair(file_name1, file_name2, verts1, faces1, verts2, faces2, 
                spectral_map=None, texture_file=None):
    """
    Write a pair of 3D meshes with texture mapping between them.
    
    Args:
        file_name1 (str): Output file path for the first mesh
        file_name2 (str): Output file path for the second mesh
        verts1 (numpy.ndarray): Vertices of the first mesh [N, 3]
        faces1 (numpy.ndarray): Faces of the first mesh [F, 3]
        verts2 (numpy.ndarray): Vertices of the second mesh [M, 3]
        faces2 (numpy.ndarray): Faces of the second mesh [G, 3]
        spectral_map (tuple or tensor): Either:
            - Tuple of (evecs_y, Cxy, evecs_trans_x) for memory-efficient mapping calculation
            - Direct p2p indices tensor
            - Sparse permutation matrix
            - Dense permutation matrix
        texture_file (str): Texture file name
    """
    # Convert tensors to numpy if needed
    if isinstance(verts1, torch.Tensor):
        verts1 = verts1.detach().cpu().numpy()
    if isinstance(faces1, torch.Tensor):
        faces1 = faces1.detach().cpu().numpy()
    if isinstance(verts2, torch.Tensor):
        verts2 = verts2.detach().cpu().numpy()
    if isinstance(faces2, torch.Tensor):
        faces2 = faces2.detach().cpu().numpy()
    
    # Generate texture coordinates for shape 1
    uv1 = generate_tex_coords(verts1)
    
    # Write obj for shape 1 if it doesn't exist
    if not os.path.exists(file_name1):
        write_obj_with_texture(verts1, faces1, file_name1, uv1, texture_file)
    
    # Map texture coords from shape 1 to shape 2 using the appropriate method
    if spectral_map is None:
        # Default identity mapping
        uv2 = uv1
    elif isinstance(spectral_map, tuple) and len(spectral_map) == 3:
        # Memory-efficient spectral mapping
        evecs_y, Cxy, evecs_trans_x = spectral_map
        
        # We need to compute: uv2 = Pyx @ uv1 where Pyx = evecs_y @ Cxy @ evecs_trans_x
        # Implement it step by step to avoid materializing the full Pyx matrix
        
        # First convert tensors to numpy for consistent processing
        if isinstance(evecs_y, torch.Tensor):
            evecs_y = evecs_y.detach().cpu().numpy()
        if isinstance(Cxy, torch.Tensor):
            Cxy = Cxy.detach().cpu().numpy()
        if isinstance(evecs_trans_x, torch.Tensor):
            evecs_trans_x = evecs_trans_x.detach().cpu().numpy()
            
        # Compute matrix multiplication in steps to avoid forming full dense matrix
        # (evecs_trans_x @ uv1) -> (Cxy @ result) -> (evecs_y @ result)
        temp = evecs_trans_x @ uv1
        temp = Cxy @ temp
        uv2 = evecs_y @ temp
    elif isinstance(spectral_map, torch.Tensor):
        if spectral_map.dim() == 1:  # p2p indices
            uv2 = uv1[spectral_map.detach().cpu().numpy()]
        elif spectral_map.is_sparse:  # sparse tensor
            # Convert to CPU for numpy operations
            spectral_map_cpu = spectral_map.detach().cpu()
            # Use sparse matrix multiplication
            if spectral_map_cpu.layout == torch.sparse_coo:
                from scipy.sparse import coo_matrix
                indices = spectral_map_cpu.indices().numpy()
                values = spectral_map_cpu.values().numpy()
                sparse_scipy = coo_matrix((values, (indices[0], indices[1])), shape=spectral_map_cpu.shape)
                uv2 = sparse_scipy.dot(uv1)
            elif spectral_map_cpu.layout == torch.sparse_csr:
                from scipy.sparse import csr_matrix
                crow_indices = spectral_map_cpu.crow_indices().numpy()
                col_indices = spectral_map_cpu.col_indices().numpy()
                values = spectral_map_cpu.values().numpy()
                sparse_scipy = csr_matrix((values, col_indices, crow_indices), shape=spectral_map_cpu.shape)
                uv2 = sparse_scipy.dot(uv1)
            else:
                raise ValueError(f"Unsupported sparse layout: {spectral_map_cpu.layout}")
        else:  # dense tensor
            spectral_map_np = spectral_map.detach().cpu().numpy()
            uv2 = spectral_map_np @ uv1
    else:  # numpy array
        uv2 = spectral_map @ uv1
    
    # Write obj for shape 2
    write_obj_with_texture(verts2, faces2, file_name2, uv2, texture_file)

#def write_obj_pair(file_name1, file_name2, verts1, faces1, verts2, faces2, Pyx, texture_file):
#    # write off for shape 1
#    uv1 = generate_tex_coords(verts1)
#    if not os.path.exists(file_name1):
#        write_obj_with_texture(verts1, faces1, file_name1, uv1, texture_file)

#    # write off for shape 2
#    uv2 = Pyx @ uv1
#    write_obj_with_texture(verts2, faces2, file_name2, uv2, texture_file)


def generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    ind = np.argsort(np.std(verts, axis=0))[::-1]
    verts = verts[:, ind]
    vt = np.stack([verts[:, col1], verts[:, col2]], axis=-1)
    vt -= np.min(vt, axis=0, keepdims=True)
    vt = mult_const * vt / np.max(vt)
    vt[:, 0] = (vt[:, 0] - vt[:, 0].min()) / (vt[:, 0].max() - vt[:, 0].min())
    vt[:, 1] = (vt[:, 1] - vt[:, 1].min()) / (vt[:, 1].max() - vt[:, 1].min())
    return vt


def write_obj_with_texture(verts, faces, file_name, uv=None, texture_name='texture.png'):
    """
    write .obj file with texture.
    Args:
        verts (np.ndarray): vertices. [V, 3].
        faces (np.ndarray): faces. [F, 3].
        uv (np.ndarray, None): texture maps. [V, 2]
        texture_name (str): texture map image file name.
        file_name (str): stored file name.
    """
    assert verts.shape[-1] == 3, f'vertex does not have the correct format: {verts.shape}.'
    assert faces.shape[-1] == 3, f'face does not have the correct format: {faces.shape}.'
    if uv is not None:
        assert uv.shape[-1] == 2, f'vertex texture does not have the correct format: {uv.shape}.'
    object_name = os.path.splitext(os.path.basename(file_name))[0]
    faces = faces.astype(int)
    with open(file_name, 'w') as f:
        # head
        f.write('# write_obj (c) 2004 Gabriel Peyr\n')
        f.write(f'mtllib ./{object_name}.mtl\n')
        f.write(f'g\n# object {object_name} to come\n')

        # vertex position
        f.write(f'# {verts.shape[0]} vertex\n')
        for i in range(verts.shape[0]):
            f.write(f'v {verts[i][0]:.6f} {verts[i][1]:.6f} {verts[i][2]:.6f}\n')

        # use mtl
        f.write(f'g {object_name}_export\n')
        mtl_bump_name = 'material_0'
        f.write(f'usemtl {mtl_bump_name}\n')

        # face
        f.write(f'# {faces.shape[0]} faces\n')
        faces += 1
        for i in range(faces.shape[0]):
            f.write(f'f {faces[i][0]}/{faces[i][0]} {faces[i][1]}/{faces[i][1]} {faces[i][2]}/{faces[i][2]}\n')

        # vertex texture
        if uv is not None:
            for i in range(uv.shape[0]):
                f.write(f'vt {uv[i][0]:.6f} {uv[i][1]:.6f}\n')
        else:
            vertext = verts[:, 0:2] * 0 - 1
            # vertex position
            f.write(f'# {vertext.shape[0]} vertex texture\n')
            for i in range(vertext.shape[0]):
                f.write(f'vt {vertext[i][0]:.6f} {vertext[i][1]:.6f}\n')

    # generate MTL file
    if uv is not None:
        mtl_file = file_name.replace('.obj', '.mtl')
        Ka = [0.2, 0.2, 0.2]
        Kd = [1, 1, 1]
        Ks = [1, 1, 1]
        Tr = 1
        Ns = 0
        illum = 2
        with open(mtl_file, 'a') as f:
            f.write('# write_obj (c) 2004 Gabriel Peyr\n')
            f.write(f'newmtl {mtl_bump_name}\n')
            f.write(f'Ka  {Ka[0]:.6f} {Ka[1]:.6f} {Ka[2]:.6f}\n')
            f.write(f'Kd  {Kd[0]:.6f} {Kd[1]:.6f} {Kd[2]:.6f}\n')
            f.write(f'Ks  {Ks[0]:.6f} {Ks[1]:.6f} {Ks[2]:.6f}\n')
            f.write(f'Tr  {Tr}\n')
            f.write(f'Ns  {Ns}\n')
            f.write(f'illum {illum}\n')
            f.write(f'map_Kd {texture_name}\n')
            f.write('#\n# EOF\n')


def create_colormap(verts):
    minx = verts[:, 0].min()
    miny = verts[:, 1].min()
    minz = verts[:, 2].min()
    maxx = verts[:, 0].max()
    maxy = verts[:, 1].max()
    maxz = verts[:, 2].max()
    r = (verts[:, 0] - minx) / (maxx - minx)
    g = (verts[:, 1] - miny) / (maxy - miny)
    b = (verts[:, 2] - minz) / (maxz - minz)
    colors = np.stack((r, g, b), axis=-1)
    assert colors.shape == verts.shape
    return colors


def write_point_cloud_pair(file_name1, file_name2, verts1, verts2, p2p):
    # create pcd for shape 1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(verts1)
    colors = create_colormap(verts1)
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    # create pcd for shape 2
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(verts2)
    colors = colors[p2p]
    pcd2.colors = o3d.utility.Vector3dVector(colors)

    # save results
    if not os.path.exists(file_name1):
        o3d.io.write_point_cloud(file_name1, pcd1)
    o3d.io.write_point_cloud(file_name2, pcd2)
