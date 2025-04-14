import trimesh, os
from tqdm import tqdm
import numpy as np
import open3d as o3d

from utils.misc import plot_with_std
from utils.tensor_util import to_numpy
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_generalization(ssm_model, dataloader_test, deformed_testing_shapes, logger, device, output_path, template):
    surface_distance = SurfaceDistanceOpen3D()

    print("OUTPUT TO ", output_path)
    
    generalizations_mean = []
    generalizations_std = []
    logger.info(f'Calculating Generalization')

    for mode in tqdm(range(1, ssm_model.variances.shape[0] + 1)):
        generalizations_per_mode = []

        for index, test_data in enumerate(dataloader_test):
            original_verts = (test_data['verts'].to(device) * test_data['face_area'].to(device)).float()
            recon_deformed_shape = (ssm_model.get_reconstruction(deformed_testing_shapes[index], n_modes=mode).reshape(1, -1, 3).to(device)).float()
            surf_dist = surface_distance(to_numpy(original_verts), to_numpy(original_verts), to_numpy(recon_deformed_shape), to_numpy(template['faces']))[0]
            generalizations_per_mode.append(surf_dist)


        generalization_per_mode_mean = np.mean(generalizations_per_mode)
        generalization_per_mode_std = np.std(generalizations_per_mode)
        generalizations_mean.append(generalization_per_mode_mean)
        generalizations_std.append(generalization_per_mode_std)
        logger.info(
            f'Generalizations for mode {mode} is {generalization_per_mode_mean:.4f} +/- {generalization_per_mode_std:.4f}')

    result_path = os.path.join(output_path, "generality.png")
    generalizations_mean = np.array(generalizations_mean)
    generalizations_std = np.array(generalizations_std)
    plot_with_std(np.array(list(range(1, ssm_model.variances.shape[0] + 1))),
                  generalizations_mean, generalizations_std,
                  "Generality in mm", result_path)
    np.save(os.path.join(output_path, "generalizations_mean.npy"), generalizations_mean)
    np.save(os.path.join(output_path, "generalizations_std.npy"), generalizations_std)


class SurfaceDistanceOpen3D():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, verticesA, facesA, verticesB, facesB):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        m1 = o3d.t.geometry.TriangleMesh()
        m2 = o3d.t.geometry.TriangleMesh()
        m1.vertex.positions = o3d.core.Tensor(verticesA.astype(np.float32))
        m1.triangle.indices = o3d.core.Tensor(facesA)
        m2.vertex.positions = o3d.core.Tensor(verticesB.astype(np.float32))
        m2.triangle.indices = o3d.core.Tensor(facesB)
        
        scene_A = o3d.t.geometry.RaycastingScene()
        _ = scene_A.add_triangles(m1)

        scene_B = o3d.t.geometry.RaycastingScene()
        _ = scene_B.add_triangles(m2)

        rA = scene_A.compute_distance(verticesB.astype(np.float32))
        rB = scene_B.compute_distance(verticesA.astype(np.float32))
        
        #_, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        #_, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5*(rA.numpy().mean() + rB.numpy().mean())
        #distance = .5 * np.array(A_B_dist).mean() + .5 * \
        #    np.array(B_A_dist).mean()
        return np.array([distance])

class SurfaceDistance():
    """This class calculates the symmetric vertex to surface distance of two
    trimesh meshes.
    """

    def __init__(self):
        pass

    def __call__(self, A, B):
        """
        Args:
          A: trimesh mesh
          B: trimesh mesh
        """
        scene_A = o3d.t.geometry.RaycastingScene()
        scene_A.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(A.as_open3d))

        scene_B = o3d.t.geometry.RaycastingScene()
        scene_B.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(B.as_open3d))

        rA = scene_A.compute_distance(B.vertices.astype(np.float32))
        rB = scene_B.compute_distance(A.vertices.astype(np.float32))
        
        #_, A_B_dist, _ = trimesh.proximity.closest_point(A, B.vertices)
        #_, B_A_dist, _ = trimesh.proximity.closest_point(B, A.vertices)
        distance = .5*(rA.numpy().mean() + rB.numpy().mean())
        #distance = .5 * np.array(A_B_dist).mean() + .5 * \
        #    np.array(B_A_dist).mean()
        print(distance)
        return np.array([distance])