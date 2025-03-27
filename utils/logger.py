# logging utils for training and validation
import datetime
import logging
import time

from .dist_util import get_dist_info, master_only
from .tensor_util import to_numpy

# initialized logger
initialized_logger = {}


class AvgTimer:
    """
    Timer to record the average elapsed time.

    Usage:
        timer = AvgTimer()
        for _ in range(100):
            timer.start()
            ... # do something
            timer.record()
            print(timer.get_current_time()) # print current elapsed time
        print(timer.get_avg_time()) # print average elapsed time
    """

    def __init__(self, window=200):
        """
        Args:
            window (int, optional): Sliding window to compute average time. Default 200.
        """
        self.window = window
        self.current_time = 0.
        self.total_time = 0.
        self.avg_time = 0.
        self.count = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.count += 1
        # calculate current time
        self.current_time = time.time() - self.start_time
        # calculate total time
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset timer
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger:
    """
    Message Logger

    Args:
        opt (dict): Config dict. It contains the following keys:
            name (str): experiment name.
            logger (dict): Contains 'print_freq' as logging interval.
            train (dict): Contains 'total_iter' as total iterations.

        start_iter (int, optional): Start iteration number. Default 1.
        tb_logger (SummaryWriter, optional): Tensorboard logger. Default None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        """
        Reset start time.
        """
        self.start_time = time.time()

    @master_only
    def __call__(self, log_dict):
        """
        Logging message using wandb instead of tensorboard
        
        Args:
            log_dict (dict): logging dictionary with the following keys:
                epoch (int): Current epoch.
                iter (int): Current iteration.
                lrs (list): List of learning rates.
                time (float): Elapsed time for one iteration.
                data_time (float): Elapsed time of data fetch for one iteration.
        """
        # epoch, iter, learning rates
        epoch = log_dict.pop('epoch')
        current_iter = log_dict.pop('iter')
        lrs = log_dict.pop('lrs')
    
        # format message
        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')]'
    
        # time and estimated time
        if 'time' in log_dict.keys():
            iter_time = log_dict.pop('time')
            data_time = log_dict.pop('data_time')
            # compute the total time
            total_time = time.time() - self.start_time
            # estimate the average time for one iteration
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            # estimate the rest time for the whole training
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            # add the estimated time to message
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})]'
    
        # prepare wandb logs
        wandb_logs = {
            'epoch': epoch,
            'iter': current_iter,
        }
        # Add learning rates
        for i, lr in enumerate(lrs):
            wandb_logs[f'lr_{i}'] = lr
        
        # Add time metrics
        if 'time' in locals():
            wandb_logs['time'] = iter_time
            wandb_logs['data_time'] = data_time
        
        # Other items, for example losses
        for k, v in log_dict.items():
            message += f'{k}: {v:.4e} '
            # Add to wandb logs
            wandb_logs[k] = v
    
        # Log to wandb
        if self.tb_logger:  # Now 'tb_logger' is actually wandb
            self.tb_logger.log(wandb_logs, step=current_iter)
    
        # print message
        self.logger.info(message)

@master_only
def log_mesh(logger, tag, vertices, faces, global_step=None, colors=None):
    """Log mesh to wandb using Plotly with improved visualization settings
    
    Args:
        logger: wandb logger
        tag (str): Name for the mesh visualization
        vertices (tensor/array): Mesh vertices
        faces (tensor/array): Mesh faces
        global_step (int, optional): Global step value
        colors (tensor/array, optional): Colors for the mesh
    """
    # Skip if logger is None
    if logger is None:
        return
        
    try:
        import numpy as np
        import plotly.graph_objects as go
        from utils.tensor_util import to_numpy
        
        # Convert tensors to numpy
        vertices_np = to_numpy(vertices)
        faces_np = to_numpy(faces)
        
        # Ensure vertices are shaped correctly [N, 3]
        if vertices_np.ndim == 3 and vertices_np.shape[0] == 1:
            vertices_np = vertices_np.squeeze(0)
        
        # Ensure faces are shaped correctly [N, 3] 
        if faces_np.ndim == 3 and faces_np.shape[0] == 1:
            faces_np = faces_np.squeeze(0)
            
        # Create color values based on vertex height for better visualization
        if colors is None:
            # Use height (z-coordinate) for coloring
            z_vals = vertices_np[:, 2]
            min_z, max_z = np.min(z_vals), np.max(z_vals)
            colors = (z_vals - min_z) / (max_z - min_z + 1e-8)
            
        # Extract coordinates for Plotly
        x, y, z = vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2]
        i, j, k = faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]
        
        # Create the 3D mesh with improved visualization settings
        fig = go.Figure()
        
        # Add the filled mesh with a soft color palette
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            intensity=colors,
            colorscale='Blues',  # Use a nicer colorscale - try 'Blues', 'RdBu', or 'Cividis'
            opacity=1.0,         # Slightly transparent to see structure
            lighting=dict(
                ambient=0.6,      # Higher ambient light
                diffuse=0.5,      # Moderate diffuse
                roughness=0.2,    # Some roughness
                specular=0.3,     # Moderate specular
                fresnel=0.01       # No fresnel
            ),
            flatshading=True,    # Try smooth shading
            hoverinfo='none',
            showscale=False       # Hide the color scale
        ))
        
        # Add mesh edges (wireframe) as line segments
        edges = set()
        for face in faces_np:
            # Add edges for each face, ensuring each edge is only added once
            edges.add(tuple(sorted([int(face[0]), int(face[1])])))
            edges.add(tuple(sorted([int(face[1]), int(face[2])])))
            edges.add(tuple(sorted([int(face[2]), int(face[0])])))
        
        # Create edge line segments
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in edges:
            # Add the first vertex
            edge_x.append(vertices_np[edge[0], 0])
            edge_y.append(vertices_np[edge[0], 1])
            edge_z.append(vertices_np[edge[0], 2])
            
            # Add the second vertex
            edge_x.append(vertices_np[edge[1], 0])
            edge_y.append(vertices_np[edge[1], 1])
            edge_z.append(vertices_np[edge[1], 2])
            
            # Add None to create separation between edges
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)
        
        # Add the edges as a scatter3d trace
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='black',
                width=1.5
            ),
            hoverinfo='none'
        ))
        
        # Enhanced layout settings
        fig.update_layout(
            scene=dict(
                aspectmode='cube',  # Equal aspect ratio
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),  # Better camera position
                    up=dict(x=0, y=0, z=1)          # Camera up direction
                ),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
                dragmode='turntable'  # Makes rotation more intuitive
            ),
            margin=dict(l=0, r=0, t=35, b=0),
            title=dict(
                text=tag,
                x=0.5,
                y=0.95,
                font=dict(size=16)
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            width=500,                      # Fixed width for better display in Wandb
            height=500                      # Fixed height for better display
        )
        
        # Log to wandb with a namespace for validation metrics to avoid step conflicts
        validation_tag = f"validation/{tag}"
        validation_step = global_step if global_step is not None else 0
        
        # Log both the visualization and the step
        logger.log({
            validation_tag: fig,
            "validation_step": validation_step
        }, step=None)  # Don't pass the step parameter here
        
    except ImportError as e:
        print(f"Warning: Missing import: {e}. Install with 'pip install plotly numpy' for mesh visualization.")
    except Exception as e:
        print(f"Warning: Failed to log mesh visualization: {e}")
        # Fallback: Just log the shape information
        logger.log({f"{tag}_info": f"Vertices: {vertices.shape}, Faces: {faces.shape}"}, step=global_step)

@master_only
def log_mesh_old(logger, tag, vertices, faces, global_step=None, colors=None):
    """Log mesh to wandb
    
    Args:
        logger: wandb logger
        tag (str): Name for the mesh visualization
        vertices (tensor): Mesh vertices
        faces (tensor): Mesh faces
        global_step (int, optional): Global step value
        colors (tensor, optional): Colors for the mesh
    """
    # Skip if logger is None
    if logger is None:
        return
        
    # For wandb, convert tensors to numpy
    from utils.tensor_util import to_numpy
    
    vertices_np = to_numpy(vertices)
    faces_np = to_numpy(faces)
    
    # Create wandb 3D object
    data = {
        "type": "mesh",
        "vertices": vertices_np,
        "faces": faces_np
    }
    
    if colors is not None:
        colors_np = to_numpy(colors)
        data["colors"] = colors_np
        
    # Log to wandb
    logger.log({tag: logger.Object3D(data)}, step=global_step)

@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

# 2. Modify the logger.py to add wandb integration
# Add to utils/logger.py
@master_only
def init_wandb_logger(opt):
    """Initialize wandb logger.
    
    Args:
        opt (dict): Config dict containing wandb settings.
        
    Returns:
        wandb object.
    """
    import wandb
    
    project_name = opt.get('wandb', {}).get('project', opt['name'])
    wandb_dir = opt.get('wandb', {}).get('dir', opt['path']['log'])
    resume_id = opt.get('wandb', {}).get('resume_id', None)
    entity = opt.get('wandb', {}).get('entity', 'karabelaselias-university-of-graz')
    
    wandb.init(
        project=project_name,
        name=f"{project_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
        entity = entity,
        id=resume_id,
        dir=wandb_dir,
        config=opt,
        resume="allow" if resume_id else None
    )
    # Define custom step metrics to handle out-of-order logging
    if wandb.run is not None:
        wandb.define_metric("validation/*", step_metric="validation_step")
    return wandb


def get_root_logger(logger_name='root_logger', log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str, optional): root logger name. Default: 'root_logger'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger. Default None.
        log_level (int, optional): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time. Default logging.INFO.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it.
    if logger_name in initialized_logger:
        return logger

    # initialize stream handler
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    # initialize logger level for each process
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # add logger to initialized logger
    initialized_logger[logger_name] = True

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    #import torchvision
    import platform

    msg = ('\nVersion Information: '
           f'\n\tPython: {platform.python_version()}'
           f'\n\tPyTorch: {torch.__version__}')
           #f'\n\tTorchVision: {torchvision.__version__}')
    return msg