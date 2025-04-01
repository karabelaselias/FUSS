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
from utils.amp_utils import disable_amp
from utils.memory_utils import profile_memory, memory_efficient_computation

import gc


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
            # get data pair
            data_x, data_y = transfer_batch_to_device(data['first'], self.device), transfer_batch_to_device(data['second'], self.device)
            assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'
         
            self.network_timers['feature_extractor'].start()
            feat_x = self.networks['feature_extractor'](data_x)  # [B, Nx, C]
            feat_y = self.networks['feature_extractor'](data_y)  # [B, Ny, C]
            self.network_timers['feature_extractor'].record()
            
            self.network_timers['permutation'].start()
            Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # [B, Nx, Ny], [B, Ny, Nx]
            self.network_timers['permutation'].record()
            
            # compute functional map related loss
            if 'surfmnet_loss' in self.losses:
                self.network_timers['fmap_net'].start()
                self.compute_fmap_loss(data_x, data_y, feat_x, feat_y, Pxy, Pyx)
                self.network_timers['fmap_net'].record()
    
            # Interpolation
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

            del vert_x_1, vert_y_1
    
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
                del shape_x_diff_arr, shape_y_diff_arr
    
            del Pxy, Pyx
            
            # shape deformation losses
            if 'dirichlet_shape_loss' in self.losses:
                Lx, Ly = data_x['L'][0], data_y['L'][0]
                self.compute_shape_interpolation_dirichlet_loss(vert_x, vert_x_pred_arr, Lx)
                self.compute_shape_interpolation_dirichlet_loss(vert_y, vert_y_pred_arr, Ly)
    
            if 'chamfer_shape_loss' or 'earth_mover_loss' in self.losses:
                self.compute_shape_interpolation_chamfer_loss(vert_x, vert_x_pred_arr, vert_y, vert_y_pred_arr,
                                                              face_x, face_y)

            if 'edge_shape_loss' in self.losses:
                self.compute_shape_edge_loss(vert_x_pred_arr, vert_y_pred_arr, face_x, face_y)
            
            self.network_timers['loss_stuff'].record()

            del vert_x_pred_arr, vert_y_pred_arr 

        


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
        elif 'earth_mover_loss' in self.losses:
            if 'l_emd' not in self.loss_metrics:
                self.loss_metrics['l_emd'] = 0.0

            if self.pose_timestep > 0:
                for tp in range(self.pose_timestep + 1):
                    self.loss_metrics['l_emd'] += self.compute_earth_mover_distance(vert_x_pred_arr[:, :, tp],
                                                                              vert_y_pred_arr[:, :, self.pose_timestep - tp])
            # shape X to vert_y[:, :, T] & shape Y to vert_x[:, :, T]
            self.loss_metrics['l_emd'] += self.compute_earth_mover_distance(vert_x, vert_y_pred_arr[:, :, self.pose_timestep])

            self.loss_metrics['l_emd'] += self.compute_earth_mover_distance(vert_x_pred_arr[:, :, self.pose_timestep], vert_y)
            

    def compute_earth_mover_distance(self, vert_x, vert_y):
        with disable_amp():
            loss_emd = self.losses['earth_mover_loss'](vert_x, vert_y)
        return loss_emd
        
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

    # Memory-optimized compute_displacement implementation
    def compute_displacement(self,vert_x, vert_y, face_x, Pxy=None, p2p_xy=None):
        n_vert_x = vert_x.shape[0]
        
        # Align vertices
        if p2p_xy is not None:
            vert_y_align = vert_y[p2p_xy]
        elif Pxy is not None:
            vert_y_align = sparse_mm(Pxy, vert_y)
        else:
            raise ValueError("Either Pxy or p2p_xy must be provided")
        
        # Calculate displacement vector
        displacement_vector = vert_y_align - vert_x
        
        # Initialize output tensor
        vert_x_pred_arr = torch.zeros(n_vert_x, 3, self.pose_timestep + 1, device=self.device)
        
        # Process each timestep individually
        for t_idx in range(self.pose_timestep + 1):
            # Calculate time value 
            t = (t_idx + 1) / (self.pose_timestep + 1)
            
            # Create input tensor
            base_input = torch.cat((
                vert_x, 
                displacement_vector,
                torch.zeros(size=(n_vert_x, 1), device=self.device)
            ), dim=1)
            
            # Create time tensor to match original behavior
            time_tensor = torch.zeros_like(base_input)
            time_tensor[:, -1] = t
            
            # Add time component
            t_input = base_input + time_tensor
            
            # Process through network
            t_displacement = self.networks['interpolator'](
                t_input.unsqueeze(0), face_x.unsqueeze(0)
            ).squeeze(0)
            
            # Apply clipping
            t_displacement = torch.clamp(t_displacement, min=-100.0, max=100.0)
            
            # Calculate vertices at this time step
            vert_x_pred_arr[:, :, t_idx] = vert_x + torch.clamp(t_displacement * t, min=-100.0, max=100.0)
        
        return vert_x_pred_arr
    
    def compute_displacement_inefficient(self, vert_x, vert_y, face_x, Pxy=None, p2p_xy=None):
        """Compute displacement field from shape x to shape y."""
        n_vert_x = vert_x.shape[0]
        
        # Pre-compute time steps once
        step_size = 1 / (self.pose_timestep + 1)
        time_steps = step_size + torch.arange(0, 1, step_size, device=self.device).unsqueeze(1).unsqueeze(2)
        time_mask = torch.zeros(7, device=self.device)
        time_mask[-1] = 1.0
        time_steps_up = time_steps * time_mask.unsqueeze(0).unsqueeze(1)
        
        # Align vertices - same as original
        if p2p_xy is not None:
            vert_y_align = vert_y[p2p_xy]
        elif Pxy is not None:
            vert_y_align = sparse_mm(Pxy, vert_y)
        else:
            raise ValueError("Either Pxy or p2p_xy must be provided")
        
        # Create inputs - same as original
        inputs = torch.cat((
            vert_x, vert_y_align - vert_x,
            torch.zeros(size=(n_vert_x, 1), device=self.device)
        ), dim=1).unsqueeze(0)
        inputs = inputs + time_steps_up
        
        # Identical to original processing
        displacements = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3), device=self.device)
        for i in range(inputs.shape[0]):
            # Add explicit cleanup every 10,000 vertices
            if i % 10000 == 0:
                torch.cuda.empty_cache()
            displacements[i] = self.networks['interpolator'](inputs[i].unsqueeze(0), face_x.unsqueeze(0)).squeeze(0)
        
        # Same operations as original
        displacements = torch.clamp(displacements, min=-100.0, max=100.0)
        vert_x_pred_arr = vert_x.unsqueeze(0) + torch.clamp(displacements * time_steps, min=-100.0, max=100.0)
        vert_x_pred_arr = vert_x_pred_arr.permute([1, 2, 0]).contiguous()
    
        # Same NaN/Inf handling
        if torch.isnan(vert_x_pred_arr).any() or torch.isinf(vert_x_pred_arr).any():
            logger = get_root_logger()
            logger.warning("NaN or Inf detected in interpolated vertices, replacing with safe values")
            vert_x_pred_arr = torch.nan_to_num(vert_x_pred_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Added memory cleanup - only difference
        del inputs, vert_y_align, displacements
        
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
            evecs_y_pb = sparse_mm(Pxy, evecs_y.squeeze())
            evecs_x_pb = sparse_mm(Pyx, evecs_x.squeeze())
                        
            Cyx_est, Cxy_est = torch.mm(evecs_trans_x.squeeze(), evecs_y_pb).unsqueeze(0), \
                               torch.mm(evecs_trans_y.squeeze(), evecs_x_pb).unsqueeze(0)

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
        with self.amp_context():
            # compute total loss
            loss = sum(v for k, v in self.loss_metrics.items() if k != 'l_total')
            # Scale loss by accumulation steps
            scaled_loss = loss / self.accumulation_steps
            # Store original loss for logging
            self.loss_metrics['l_total'] = loss
        
        # backward pass with gradient scaling
        self.scaler.scale(scaled_loss).backward()
        
        self.accumulation_counter += 1
        
        # Only update weights after accumulation_steps
        if self.accumulation_counter >= self.accumulation_steps:
            # clip gradient for stability
            if self.opt.get('clip_grad', True):
                for key in self.optimizers:
                    self.scaler.unscale_(self.optimizers[key])
                for key in self.networks:
                    torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)
            
            # update weights
            for name in self.optimizers:
                self.scaler.step(self.optimizers[name])
                self.optimizers[name].zero_grad(set_to_none=True)
            
            # Update scaler for next iteration
            self.scaler.update()
            self.accumulation_counter = 0
        
        
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
        with disable_amp():
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
            del vert_x_pred_arr, vert_y_pred_arr
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
        
        with disable_amp():
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
