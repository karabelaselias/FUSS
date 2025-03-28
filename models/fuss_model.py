import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.logger import get_root_logger
from utils.tensor_util import to_device, to_numpy, transfer_batch_to_device
from utils.tensor_util import to_device, to_numpy
#from networks.permutation_network import apply_sparse_similarity, compute_permutation_matrix_sparse
#from networks.permutation_network import SparseSimilarity
from utils.fmap_util import fmap2pointmap_keops
from utils.torch_sparse_mm import sparse_mm
import gc


def print_memory_stats(label):
    print(f"\n=== {label} ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

@MODEL_REGISTRY.register()
class FussModel(BaseModel):
    def __init__(self, opt):
        self.pose_timestep = opt.get('pose_timestep', 0)
        self.pose_milestones = opt.get('pose_milestones', None)
        super(FussModel, self).__init__(opt)

    def update_model_per_epoch(self):
        super(FussModel, self).update_model_per_epoch()
        if self.pose_milestones is not None and self.curr_epoch in self.pose_milestones:
            logger = get_root_logger()
            logger.info(f'Update number of pose steps from '
                        f'{self.pose_timestep} to {self.pose_milestones[self.curr_epoch]}')
            self.pose_timestep = self.pose_milestones[self.curr_epoch]

    def feed_data(self, data):
        """process data with PyTorch profiling"""
        from utils.logger import AvgTimer

        # Initialize timers
        if not hasattr(self, 'network_timers'):
            self.network_timers = {
                'feature_extractor': AvgTimer(),
                'permutation': AvgTimer(),
                'fmap_net': AvgTimer(),
                'interpolator': AvgTimer(),
                'loss_stuff': AvgTimer()
            }
        
        with self.amp_context():
            #print_memory_stats("Start of feed_data")
            # get data pair
            data_x, data_y = transfer_batch_to_device(data['first'], self.device), transfer_batch_to_device(data['second'], self.device)
            assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'
            #print(f"Vertex shapes: x={data_x['verts'].shape}, y={data_y['verts'].shape}")
            
            # extract feature
            #feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
            #feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

            self.network_timers['feature_extractor'].start()
            feat_x = self.networks['feature_extractor'](data_x)  # [B, Nx, C]
            feat_y = self.networks['feature_extractor'](data_y)  # [B, Ny, C]
            self.network_timers['feature_extractor'].record()
            
            #print(f"Feature shapes: x={feat_x.shape}, y={feat_y.shape}")
            #print_memory_stats("After feature extraction")
    
            #print_memory_stats("Before permutation computation")
            self.network_timers['permutation'].start()
            Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # [B, Nx, Ny], [B, Ny, Nx]
            #print(f"Permutation matrix shapes: Pxy={Pxy.shape}, nnz={Pxy._nnz()}")
            #print_memory_stats("After permutation computation")
            self.network_timers['permutation'].record()
            
            # compute functional map related loss
            if 'surfmnet_loss' in self.losses:
                self.network_timers['fmap_net'].start()
                self.compute_fmap_loss(data_x, data_y, feat_x, feat_y, Pxy, Pyx)
                self.network_timers['fmap_net'].record()
    
            # Interpolation
            #Pxy, Pyx = Pxy.squeeze(0), Pyx.squeeze(0)
            vert_x, vert_y = data_x['verts'].squeeze(0), data_y['verts'].squeeze(0)
            face_x, face_y = data_x['faces'].squeeze(0), data_y['faces'].squeeze(0)
    
            # from shape x to shape y
            self.network_timers['interpolator'].start()
            vert_x_pred_arr = self.compute_displacement(vert_x, vert_y, face_x, Pxy)  # [n_vert_x, 3, T+1]
    
            # from shape y to shape x
            vert_y_pred_arr = self.compute_displacement(vert_y, vert_x, face_y, Pyx)  # [n_vert_y, 3, T+1] 
            
            # compute alignment loss
            vert_y_1 = sparse_mm(Pxy, vert_y_pred_arr[:, :, self.pose_timestep])
            vert_x_1 = sparse_mm(Pyx, vert_x_pred_arr[:, :, self.pose_timestep])
            self.network_timers['interpolator'].record()

            self.network_timers['loss_stuff'].start()
            #Pyx.pull_back(vert_x_pred_arr[:, :, self.pose_timestep])
            #Pyx @ vert_x_pred_arr[:, :, self.pose_timestep]
            self.compute_alignment_loss(vert_x, vert_y, vert_x_1, vert_y_1)
    
            # compute smoothness regularisation for point map
            if 'smoothness_loss' in self.losses:
                Lx, Ly = data_x['L'][0], data_y['L'][0] # this are lists
                self.compute_smoothness_loss(Pxy, Pyx, Lx, Ly, vert_x, vert_y)
    
            if self.pose_timestep > 0 and 'symmetry_loss' in self.losses:
                # [T+1, n_vert_x, 3]
                shape_x_diff_arr = self.compute_interpolation_difference(vert_x_pred_arr, vert_y_pred_arr, Pxy)
                # [T+1, n_vert_y, 3]
                shape_y_diff_arr = self.compute_interpolation_difference(vert_y_pred_arr, vert_x_pred_arr, Pyx)
    
                # compute symmetry loss
                self.compute_symmetry_loss(shape_x_diff_arr, shape_y_diff_arr)
    
            #del Pxy, Pyx
            
            # shape deformation losses
            if 'dirichlet_shape_loss' in self.losses:
                Lx, Ly = data_x['L'][0], data_y['L'][0]
                self.compute_shape_interpolation_dirichlet_loss(vert_x, vert_x_pred_arr, Lx)
                self.compute_shape_interpolation_dirichlet_loss(vert_y, vert_y_pred_arr, Ly)
    
            if 'chamfer_shape_loss' in self.losses:
                self.compute_shape_interpolation_chamfer_loss(vert_x, vert_x_pred_arr, vert_y, vert_y_pred_arr,
                                                              face_x, face_y)
    
            if 'edge_shape_loss' in self.losses:
                self.compute_shape_edge_loss(vert_x_pred_arr, vert_y_pred_arr, face_x, face_y)
            
            self.network_timers['loss_stuff'].record()

        


    def compute_shape_interpolation_dirichlet_loss(self, vert_x, vert_x_pred_arr, Lx):
        if 'l_dir' not in self.loss_metrics:
            self.loss_metrics['l_dir'] = 0.0

        self.loss_metrics['l_dir'] += self.losses['dirichlet_shape_loss'](vert_x.unsqueeze(0)
                                                                          - vert_x_pred_arr[:, :, 0].unsqueeze(0),
                                                                          Lx)
        if self.pose_timestep > 0:
            if 'l_dir_interim' not in self.loss_metrics:
                self.loss_metrics['l_dir_interim'] = 0.0
            for tp in range(self.pose_timestep):
                self.loss_metrics['l_dir_interim'] += self.losses['dirichlet_shape_loss'](vert_x_pred_arr[:, :, tp].unsqueeze(0)
                                                                                  - vert_x_pred_arr[:, :, tp + 1].unsqueeze(0),
                                                                                  Lx)

    def compute_shape_interpolation_chamfer_loss(self, vert_x, vert_x_pred_arr, vert_y, vert_y_pred_arr, face_x, face_y):
        if 'chamfer_shape_loss' in self.losses:
            if 'l_cd' not in self.loss_metrics:
                self.loss_metrics['l_cd'] = 0.0

            if self.pose_timestep > 0:
                for tp in range(self.pose_timestep + 1):
                    self.loss_metrics['l_cd'] += self.compute_chamfer_distance(vert_x_pred_arr[:, :, tp],
                                                                              vert_y_pred_arr[:, :, self.pose_timestep - tp],
                                                                              face_x, face_y)
            # shape X to vert_y[:, :, T] & shape Y to vert_x[:, :, T]
            self.loss_metrics['l_cd'] += self.compute_chamfer_distance(vert_x, vert_y_pred_arr[:, :, self.pose_timestep],
                                                                          face_x, face_y)

            self.loss_metrics['l_cd'] += self.compute_chamfer_distance(vert_x_pred_arr[:, :, self.pose_timestep],
                                                                      vert_y, face_x, face_y)

    def compute_chamfer_distance(self, vert_x, vert_y, face_x, face_y):
        # Check and fix NaN/Inf values
        if torch.isnan(vert_x).any() or torch.isinf(vert_x).any():
            logger = get_root_logger()
            logger.warning("NaN or Inf detected in vert_x, replacing with safe values")
            vert_x = torch.nan_to_num(vert_x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(vert_y).any() or torch.isinf(vert_y).any():
            logger = get_root_logger()
            logger.warning("NaN or Inf detected in vert_y, replacing with safe values")
            vert_y = torch.nan_to_num(vert_y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        mesh_x = Meshes(verts=[vert_x], faces=[face_x])
        mesh_y = Meshes(verts=[vert_y], faces=[face_y])

        # Sample points with error handling
        try:
            sample_x = sample_points_from_meshes(mesh_x, 20000)
            sample_y = sample_points_from_meshes(mesh_y, 20000)
        except ValueError as e:
            logger = get_root_logger()
            logger.error(f"Error during point sampling: {e}")
            # Return a zero loss as fallback
            return torch.tensor(0.0, device=vert_x.device, dtype=vert_x.dtype)

        loss_chamfer = self.losses['chamfer_shape_loss'](sample_x, sample_y)
        # Check for NaN loss
        if torch.isnan(loss_chamfer) or torch.isinf(loss_chamfer):
            logger = get_root_logger()
            logger.warning("NaN or Inf detected in chamfer loss, replacing with zero")
            loss_chamfer = torch.tensor(0.0, device=vert_x.device, dtype=vert_x.dtype)
        
        return loss_chamfer

    def compute_shape_edge_loss(self, vert_x_pred_arr, vert_y_pred_arr, face_x, face_y):
        if 'edge_shape_loss' in self.losses:
            if 'l_edge' not in self.loss_metrics:
                self.loss_metrics['l_edge'] = 0.0

            self.loss_metrics['l_edge'] += self.losses['edge_shape_loss'](
                                                vert_x_pred_arr[:, :, self.pose_timestep], face_x)
            self.loss_metrics['l_edge'] += self.losses['edge_shape_loss'](
                                                vert_y_pred_arr[:, :, self.pose_timestep], face_y)


    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False):
        """
        Compute permutation matrix using either dense or sparse approach based on input size.
        """
        # Check if we're using ScalableSimilarity or original Similarity
        Pxy = self.networks['sparse_permutation'](feat_x, feat_y)
        
        if bidirectional:
            Pyx = self.networks['sparse_permutation'](feat_y, feat_x)
            return Pxy, Pyx

        return Pxy
       
    def compute_displacement(self, vert_x, vert_y, face_x, Pxy=None, p2p_xy=None):
        """Compute displacement field from shape x to shape y.
        
        Args:
            vert_x: Vertices of shape x [n_vert_x, 3]
            vert_y: Vertices of shape y [n_vert_y, 3]
            face_x: Faces of shape x [n_face_x, 3]
            Pxy: Permutation matrix or (scores, indices) tuple from x to y
            p2p_xy: Point-to-point map indices
            
        Note:
            If p2p_xy is provided, it is used directly.
            Otherwise, Pxy is used to compute the alignment.
        """
        n_vert_x, n_vert_y = vert_x.shape[0], vert_y.shape[0]
    
        # Efficiently compute time steps (can be pre-computed once)
        step_size = 1 / (self.pose_timestep + 1)
        time_steps = step_size + torch.arange(0, 1, step_size, device=self.device).unsqueeze(1).unsqueeze(2)
        time_mask = torch.zeros(7, device=self.device)
        time_mask[-1] = 1.0
        time_steps_up = time_steps * time_mask.unsqueeze(0).unsqueeze(1)
        
        # Determine how to align vertices - maintain original logic
        if p2p_xy is not None:
            # Use point-to-point map directly
            vert_y_align = vert_y[p2p_xy]
        elif Pxy is not None:
            # Use permutation matrix
            #vert_y_align = apply_sparse_similarity(Pxy, vert_y.unsqueeze(0))
            vert_y_align = sparse_mm(Pxy, vert_y)
            #print(vert_y_align.shape)
        else:
            raise ValueError("Either Pxy or p2p_xy must be provided")
        
        # Rest of the function remains the same
        inputs = torch.cat((
            vert_x, vert_y_align - vert_x,
            torch.zeros(size=(n_vert_x, 1), device=self.device)
        ), dim=1).unsqueeze(0)
        inputs = inputs + time_steps_up
    
        # [n_vert_x, 3, T+1]
        displacements = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3), device=self.device)
        for i in range(inputs.shape[0]):
            displacements[i] = self.networks['interpolator'](inputs[i].unsqueeze(0), face_x.unsqueeze(0)).squeeze(0)

        # Add gradient clipping to prevent extreme values
        displacements = torch.clamp(displacements, min=-100.0, max=100.0)
        # Stable computation for interpolation
        vert_x_pred_arr = vert_x.unsqueeze(0) + torch.clamp(displacements * time_steps, min=-100.0, max=100.0)
        vert_x_pred_arr = vert_x_pred_arr.permute([1, 2, 0]).contiguous()

        # Check for NaN/Inf
        if torch.isnan(vert_x_pred_arr).any() or torch.isinf(vert_x_pred_arr).any():
            logger = get_root_logger()
            logger.warning("NaN or Inf detected in interpolated vertices, replacing with safe values")
            vert_x_pred_arr = torch.nan_to_num(vert_x_pred_arr, nan=0.0, posinf=1e6, neginf=-1e6)
    
        return vert_x_pred_arr

    def compute_interpolation_difference(self, vert_x_pred_arr, vert_y_pred_arr, Pxy):
        """Compute the difference between interpolation trajectories.
        
        Args:
            vert_x_pred_arr: Predicted vertices array for shape x [n_vert_x, 3, T+1]
            vert_y_pred_arr: Predicted vertices array for shape y [n_vert_y, 3, T+1]
            Pxy: Permutation matrix or (scores, indices) tuple from x to y
            
        Returns:
            shape_x_diff_arr: Differences between trajectories [T, n_vert_x, 3]
        """
        n_vert_x = vert_x_pred_arr.shape[0]
        shape_x_diff_arr = torch.zeros(self.pose_timestep, n_vert_x, 3, device=self.device)
        
        # Handle both dense and sparse matrix formats
        for i in range(self.pose_timestep):
                # Use sparse matrix multiplication
                vert_y_aligned = sparse_mm(Pxy, vert_y_pred_arr[:, :, self.pose_timestep - 1 - i])
                shape_x_diff_arr[i] = vert_x_pred_arr[:, :, i] - vert_y_aligned
        
        return shape_x_diff_arr

    def compute_fmap_loss(self, data_x, data_y, feat_x, feat_y, Pxy, Pyx):
        # get spectral operators
        evals_x, evals_y = data_x['evals'], data_y['evals']
        evecs_x, evecs_y = data_x['evecs'], data_y['evecs']
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'], data_y['evecs_trans']

        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

        if 'couple_loss' in self.losses:
            # Check if using sparse format
            evecs_y_pb = sparse_mm(Pxy, evecs_y.squeeze()).unsqueeze(0) #apply_sparse_similarity(Pxy, evecs_y)
            evecs_x_pb = sparse_mm(Pyx, evecs_x.squeeze()).unsqueeze(0)
                        
            Cyx_est, Cxy_est = torch.bmm(evecs_trans_x, evecs_y_pb), \
                               torch.bmm(evecs_trans_y, evecs_x_pb)

            self.loss_metrics['l_couple'] = self.losses['couple_loss'](Cxy, Cxy_est) + \
                                         self.losses['couple_loss'](Cyx, Cyx_est)

    def compute_symmetry_loss(self, shape_x_diff_arr, shape_y_diff_arr):
        self.loss_metrics['l_sym'] = (self.losses['symmetry_loss'](shape_x_diff_arr) +
                                      self.losses['symmetry_loss'](shape_y_diff_arr))

    def compute_alignment_loss(self, vert_x, vert_y, vert_x_1, vert_y_1):
        self.loss_metrics['l_align'] = self.losses['align_loss'](vert_x, vert_y_1) + \
                                       self.losses['align_loss'](vert_y, vert_x_1)

    def compute_smoothness_loss(self, Pxy, Pyx, Lx, Ly, vert_x, vert_y):
        #print(vert_y.shape)
        if 'smoothness_loss' in self.losses:
            #vert_x, vert_y = vert_x.unsqueeze(0), vert_y.unsqueeze(0)
            vert_y_pb = sparse_mm(Pxy, vert_y)
            vert_x_pb = sparse_mm(Pyx, vert_x)
            #Pxy, Pyx = Pxy.unsqueeze(0), Pyx.unsqueeze(0)
            self.loss_metrics['l_smooth'] = (self.losses['smoothness_loss'](vert_y_pb.unsqueeze(0), Lx) +
                                            self.losses['smoothness_loss'](vert_x_pb.unsqueeze(0), Ly))

    def get_timing_metrics(self):
        """Get timing metrics for wandb logging"""
        if hasattr(self, 'network_timers'):
            return {f'network_time/{k}': v.get_avg_time() for k, v in self.network_timers.items()}
        return {}
        
    def optimize_parameters(self):
        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad(set_to_none=True)
        
        # Using autocast for mixed-precision training
        with self.amp_context():
            # compute total loss
            loss = sum(v for k, v in self.loss_metrics.items() if k != 'l_total')
            # update loss metrics
            self.loss_metrics['l_total'] = loss
    
        
        # backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        # backward pass
        #loss.backward()

        # clip gradient for stability - FIX HERE:
        if self.opt.get('clip_grad', True):
            # Only unscale optimizers that exist
            for name in self.optimizers:
                self.scaler.unscale_(self.optimizers[name])
                
            # Then clip gradients for all networks
            for key in self.networks:
                torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)
    
        # update weight
        for name in self.optimizers:
            self.scaler.step(self.optimizers[name])

        # Update scaler for next iteration
        self.scaler.update()
        
    def validate_single(self, data, timer, tb_logger, index):
        # get data pair
        data_x, data_y = transfer_batch_to_device(data['first'], self.device), transfer_batch_to_device(data['second'], self.device)
        vert_x, face_x = data_x['verts'], data_x['faces']
        vert_y, face_y = data_y['verts'], data_y['faces']

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze() # get rid of the Batch dim
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()

        # start record
        timer.start()

        # feature extractor
        #feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        #feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        feat_x = self.networks['feature_extractor'](data_x)
        feat_y = self.networks['feature_extractor'](data_y)

        Pxy_sparse, Pyx_sparse = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        temp_y = sparse_mm(Pxy_sparse, evecs_y) #apply_sparse_similarity(Pxy, evecs_y)
        temp_x = sparse_mm(Pyx_sparse, evecs_x) #apply_sparse_similarity(Pyx, evecs_x)
        Cxy = evecs_trans_y @ temp_x #(Pyx @ evecs_x)
        Cyx = evecs_trans_x @ temp_y #(Pxy @ evecs_y)
        
        # convert functional map to point-to-point map
        p2p_yx = fmap2pointmap_keops(Cxy, evecs_x, evecs_y)
        p2p_xy = fmap2pointmap_keops(Cyx, evecs_y, evecs_x)

        if tb_logger is not None:
            # [n_vert_x, 3, T+1]
            vert_x_pred_arr = self.compute_displacement(vert_x.squeeze(0), vert_y.squeeze(0),
                                                        face_x.squeeze(0), Pxy_sparse, p2p_xy)
            # [n_vert_y, 3, T+1]
            vert_y_pred_arr = self.compute_displacement(vert_y.squeeze(0), vert_x.squeeze(0),
                                                        face_y.squeeze(0), Pyx_sparse, p2p_yx)

        # compute Pyx from functional map
        Cxy = evecs_trans_y @ evecs_x[p2p_yx]
        #Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # save the visualization
        if tb_logger is not None and index % 60 == 1:
            from utils.logger import log_mesh
            step = self.curr_iter // self.opt['val']['val_freq']
            # Log original mesh
            log_mesh(tb_logger, f'{index}/{0}', vertices=vert_x, faces=face_x, global_step=step)
            
            #tb_logger.add_mesh(f'{index}/{0}', vertices=vert_x, faces=face_x, global_step=step)

            # Add topo transfer and original
            vert_y_align = vert_y[0][p2p_xy]
            log_mesh(tb_logger, f'{index}/{self.pose_timestep + 2}',
                     vertices=vert_y_align.unsqueeze(0),
                     faces=face_x, global_step=step)
            log_mesh(tb_logger, f'{index}/{self.pose_timestep + 3}', 
                     vertices=vert_y, faces=face_y,
                     global_step=step)

            # add topo transfer and original
            #vert_y_align = vert_y[0][p2p_xy]
            #tb_logger.add_mesh(f'{index}/{self.pose_timestep + 2}',
            #                   vertices=vert_y_align.unsqueeze(0),
            #                   faces=face_x, global_step=step)
            #tb_logger.add_mesh(f'{index}/{self.pose_timestep + 3}', vertices=vert_y, faces=face_y,
            #                   global_step=step)

            for i in range(self.pose_timestep + 1):
                point_pred = vert_x_pred_arr[..., i].unsqueeze(0)
                log_mesh(tb_logger, f'{index}/{i + 1}', vertices=point_pred, faces=face_x, global_step=step)
                #tb_logger.add_mesh(f'{index}/{i + 1}', vertices=point_pred, faces=face_x,
                #                   global_step=step)
        # Return p2p_yx, spectral components (evecs_y, Cxy, evecs_trans_x), and Cxy
        spectral_components = (evecs_y, Cxy, evecs_trans_x)
        
        return p2p_yx, spectral_components, Cxy

    @torch.no_grad()
    def get_loss_between_shapes(self, data):
        data_x, data_y = transfer_batch_to_device(data['first'], self.device), transfer_batch_to_device(data['second'], self.device)
        assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'

        # extract feature
        #feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        #feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        feat_x = self.networks['feature_extractor'](data_x)  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y)  # [B, Ny, C]

        
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # [B, Nx, Ny], [B, Ny, Nx]

        # compute functional map related loss
        self.compute_fmap_loss(data_x, data_y, feat_x, feat_y, Pxy, Pyx)

        total = 0
        for k, v in self.loss_metrics.items():
            total += v

        return data_x['name'][0], total


    @torch.no_grad()
    def deform_template(self, data):
        data_t, data_x = transfer_batch_to_device(self.template, self.device), transfer_batch_to_device(data, self.device)
        name_t, name_x = data_t['name'], data_x['name'][0]

        # get spectral operators
        evecs_t = data_t['evecs'].squeeze()
        evecs_x = data_x['evecs'].squeeze()
        evecs_trans_t = data_t['evecs_trans'].squeeze()

        # extract feature
        feat_t = self.networks['feature_extractor'](data_t['verts'].unsqueeze(0), data_t['faces'].unsqueeze(0))
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Ny, C]

        vert_t, vert_x = data_t['verts'], data_x['verts'].squeeze(0)
        face_t, face_x = data_t['faces'], data_x['faces'].squeeze(0)

        # Interpolation
        Ptx = self.compute_permutation_matrix(feat_t, feat_x, bidirectional=False)  # [B, Nx, Ny], [B, Ny, Nx]
        temp_tx = sparse_mm(Ptx, evecs_x.squeeze(0))
        Cxt = evecs_trans_t @ temp_tx.unsqueeze(0) #(Ptx @ evecs_x)
        # convert functional map to point-to-point map
        p2p_tx = fmap2pointmap_keops(Cxt, evecs_x, evecs_t)

        # from template to shape x
        vert_x_pred_arr = self.compute_displacement(vert_t, vert_x, face_t, None, p2p_tx)

        deformed_verts = vert_x_pred_arr[:, :, self.pose_timestep] * data_x['face_area']

        return deformed_verts, name_x

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'sparse_permutation' in self.networks:
            old_k = self.networks['sparse_permutation'].k_neighbors
            self.networks['sparse_permutation'].k_neighbors = 1
            self.networks['sparse_permutation'].hard = False
            #self.networks['sparse_permutation'].use_streams = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(FussModel, self).validation(dataloader, tb_logger, update)
        if 'sparse_permutation' in self.networks:
            self.networks['sparse_permutation'].k_neighbors = old_k
            self.networks['sparse_permutation'].hard = False
            #self.networks['sparse_permutation'].use_streams = self.networks['sparse_permutation'].streams > 1
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
