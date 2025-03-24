import torch
import time
import gc
import os

# Use pynvml for reliable memory tracking
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    print("pynvml not found. Install with 'pip install pynvml' for accurate GPU memory tracking.")

def get_mask(evals1, evals2, resolvant_gamma):
    """Compute mask for functional map regularization"""
    masks = []
    for bs in range(evals1.shape[0]):
        scaling_factor = max(torch.max(evals1[bs]), torch.max(evals2[bs]))
        evals1_norm, evals2_norm = evals1[bs] / scaling_factor, evals2[bs] / scaling_factor
        evals_gamma1 = (evals1_norm ** resolvant_gamma)[None, :]
        evals_gamma2 = (evals2_norm ** resolvant_gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        masks.append(M_re.square() + M_im.square())
    return torch.stack(masks, dim=0)

# Original implementation from fmap_network.py
class FMNetOriginal(torch.nn.Module):
    def __init__(self, lmbda=100, resolvant_gamma=0.5):
        super(FMNetOriginal, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """Original implementation using matrix inverse"""
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            # Create diagonal matrices one by one for each batch
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) 
                             for bs in range(evals_x.shape[0])], dim=0)
            
            # Use matrix inverse followed by multiplication
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), 
                          B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy

# Optimized implementation
class FMNetOptimized(torch.nn.Module):
    def __init__(self, lmbda=100, resolvant_gamma=0.5):
        super(FMNetOptimized, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """Optimized implementation using linear solve and chunking"""
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        B_size, K, _ = A.shape
        _, num_points_x, _ = feat_x.shape  # Extract number of points
        _, num_points_y, _ = feat_y.shape
        num_points = max(num_points_x, num_points_y)
        
        Cxy = torch.zeros((B_size, K, K), device=feat_x.device, dtype=feat_x.dtype)
        
        # Process in chunks to reduce memory usage
        # Adaptive chunk sizing for optimal performance
        if num_points < 10000:  # Small scale
            chunk_size = min(32, K)
        elif num_points < 100000:  # Medium scale
            chunk_size = min(64, K)
        else:  # Large scale
            chunk_size = K  # No chunking for large scale

        for chunk_start in range(0, K, chunk_size):
            chunk_end = min(chunk_start + chunk_size, K)
            
            # Pre-allocate a batch of diagonal matrices for this chunk
            D_batch = torch.zeros((B_size, chunk_end-chunk_start, K, K), device=A_A_t.device, dtype=A_A_t.dtype)
            
            # Fill diagonal matrices efficiently for all batch items and chunk rows
            for b in range(B_size):
                for i_rel, i_abs in enumerate(range(chunk_start, chunk_end)):
                    D_batch[b, i_rel].diagonal().copy_(D[b, i_abs])
            
            # Stack the system matrices and right-hand sides
            systems = A_A_t.unsqueeze(1) + self.lmbda * D_batch  # [B, chunk_size, K, K]
            rhs = B_A_t[:, chunk_start:chunk_end].transpose(2, 1).unsqueeze(-1)  # [B, K, chunk_size, 1]
            
            # Reshape for batched solve
            systems_flat = systems.reshape(-1, K, K)  # [B*chunk_size, K, K]
            rhs_flat = rhs.permute(0, 2, 1, 3).reshape(-1, K, 1)  # [B*chunk_size, K, 1]
            
            # Solve all systems in the chunk at once
            C_flat = torch.linalg.solve(systems_flat, rhs_flat)
            
            # Reshape back and store results
            C_chunk = C_flat.reshape(B_size, chunk_end-chunk_start, K, 1)
            Cxy[:, chunk_start:chunk_end] = C_chunk.squeeze(-1)
            
        return Cxy

# Reliable GPU memory measurement
def get_gpu_memory_usage(device_id=0):
    """Get GPU memory usage in MB using multiple methods for reliability"""
    if NVML_AVAILABLE:
        try:
            # Initialize NVML each time
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory = info.used / (1024 ** 2)
            return memory
        except Exception as e:
            pass
    
    # Fallback: use nvidia-smi through subprocess
    try:
        import subprocess
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader', 
                f'-i', f"{device_id}"
            ], 
            encoding='utf-8'
        )
        return float(result.strip())
    except Exception as e:
        print(f"nvidia-smi error: {e}")
        # Last resort: use PyTorch's built-in tracking
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

def measure_performance(model, *args, device_id=0):
    """Measure with direct nvidia-smi calls"""
    import subprocess

    def get_gpu_memory():
        result = subprocess.check_output(
            [
                'nvidia-smi', f'--id={device_id}',
                '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], 
            encoding='utf-8'
        )
        return int(result.strip())
    
    # Force reset and prepare
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    # Baseline memory
    base_memory = get_gpu_memory()
    print(f"Base memory: {base_memory} MB")
    
    # Run model with new tensors
    new_args = [arg.clone().detach().requires_grad_(True) if isinstance(arg, torch.Tensor) and arg.requires_grad else arg for arg in args]
    
    # Forward pass (force execution)
    start_time = time.time()
    output = model(*new_args)
    torch.cuda.synchronize()
    forward_time = time.time() - start_time
    
    # Measure peak memory after forward
    forward_memory = get_gpu_memory() - base_memory
    print(f"Forward memory: {forward_memory} MB")
    
    # Backward pass
    loss = output.sum()
    start_time = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start_time
    
    # Measure peak memory after backward
    backward_memory = get_gpu_memory() - forward_memory - base_memory
    print(f"Backward memory: {backward_memory} MB")
    
    return {
        'output': output.detach(),
        'forward_time': forward_time,
        'forward_memory': forward_memory,
        'backward_time': backward_time,
        'backward_memory': backward_memory,
        'gradients': [arg.grad for arg in new_args if isinstance(arg, torch.Tensor) and arg.requires_grad]
    }

# Performance measurement function
def measure_performance_old(model, *args, device_id=0):
    """Measure forward and backward performance"""
    # Clean up memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    # Record baseline memory
    baseline_memory = get_gpu_memory_usage(device_id)
    print(f"Baseline GPU memory: {baseline_memory:.2f} MB")
    
    # Clone inputs for gradient calculation
    new_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            new_arg = arg.clone().detach().requires_grad_(True)
        else:
            new_arg = arg
        new_args.append(new_arg)
    
    # Measure forward pass
    torch.cuda.synchronize()
    start_time = time.time()
    output = model(*new_args)
    torch.cuda.synchronize()
    forward_time = time.time() - start_time
    
    # Record forward memory
    forward_memory = get_gpu_memory_usage(device_id)
    forward_memory_used = forward_memory - baseline_memory
    print(f"Forward pass GPU memory: {forward_memory:.2f} MB (Used: {forward_memory_used:.2f} MB)")
    
    # Reset baseline for backward pass
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    backward_baseline = get_gpu_memory_usage(device_id)
    
    # Measure backward pass
    loss = output.mean()
    torch.cuda.synchronize()
    start_time = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start_time
    
    # Record backward memory
    backward_memory = get_gpu_memory_usage(device_id)
    backward_memory_used = backward_memory - backward_baseline
    print(f"Backward pass GPU memory: {backward_memory:.2f} MB (Used: {backward_memory_used:.2f} MB)")
    
    # Extract gradients
    gradients = []
    for arg in new_args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            gradients.append(arg.grad.clone() if arg.grad is not None else None)
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return {
        'output': output.detach(),
        'forward_time': forward_time,
        'forward_memory': forward_memory_used,
        'backward_time': backward_time,
        'backward_memory': backward_memory_used,
        'gradients': gradients
    }

def compare_implementations(batch_size, num_points, feat_dim, k_eig, device="cuda", device_id=0):
    """Compare original and optimized implementations"""
    print(f"\nTesting with batch_size={batch_size}, num_points={num_points}, feat_dim={feat_dim}, k_eig={k_eig}")
    
    # Generate random test data
    torch.manual_seed(42)
    feat_x = torch.randn(batch_size, num_points, feat_dim, device=device, requires_grad=True)
    feat_y = torch.randn(batch_size, num_points, feat_dim, device=device, requires_grad=True)
    
    # Generate random eigenvalues (sorted for realism)
    evals_x = torch.sort(torch.rand(batch_size, k_eig, device=device))[0]
    evals_y = torch.sort(torch.rand(batch_size, k_eig, device=device))[0]
    
    # Generate random eigenvector pseudo-inverses
    evecs_trans_x = torch.randn(batch_size, k_eig, num_points, device=device) / (num_points ** 0.5)
    evecs_trans_y = torch.randn(batch_size, k_eig, num_points, device=device) / (num_points ** 0.5)
    
    # Create models
    original_model = FMNetOriginal().to(device)
    optimized_model = FMNetOptimized().to(device)
    
    # Measure performance for original model
    print("\nTesting original implementation...")
    orig_results = measure_performance(
        original_model, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, device_id=device_id
    )
    
    # Measure performance for optimized model
    print("\nTesting optimized implementation...")
    opt_results = measure_performance(
        optimized_model, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, device_id=device_id
    )
    
    # Check outputs match
    outputs_match = torch.allclose(
        orig_results['output'], opt_results['output'], rtol=1e-4, atol=1e-4
    )
    
    # Calculate relative differences
    rel_diff = torch.abs((orig_results['output'] - opt_results['output']) / 
                         (orig_results['output'] + 1e-8))
    
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    
    # Check gradients match
    grads_match = all(
        g1 is None and g2 is None or 
        (g1 is not None and g2 is not None and torch.allclose(g1, g2, rtol=1e-4, atol=1e-4))
        for g1, g2 in zip(orig_results['gradients'], opt_results['gradients'])
    )
    
    # Print performance comparison
    print("\n=== Results ===")
    print(f"Outputs match: {outputs_match}")
    print(f"  Max relative difference: {max_rel_diff:.6f}")
    print(f"  Mean relative difference: {mean_rel_diff:.6f}")
    print(f"Gradients match: {grads_match}")
    
    print("\n=== Forward Pass ===")
    print(f"Original:  {orig_results['forward_time']:.4f}s, {orig_results['forward_memory']:.2f} MB")
    print(f"Optimized: {opt_results['forward_time']:.4f}s, {opt_results['forward_memory']:.2f} MB")
    print(f"Speedup: {orig_results['forward_time'] / opt_results['forward_time']:.2f}x")
    
    if opt_results['forward_memory'] > 0.01:
        memory_ratio = orig_results['forward_memory'] / opt_results['forward_memory']
        print(f"Memory reduction: {memory_ratio:.2f}x")
    else:
        print("Memory usage too small to measure accurately")
    
    print("\n=== Backward Pass ===")
    print(f"Original:  {orig_results['backward_time']:.4f}s, {orig_results['backward_memory']:.2f} MB")
    print(f"Optimized: {opt_results['backward_time']:.4f}s, {opt_results['backward_memory']:.2f} MB")
    print(f"Speedup: {orig_results['backward_time'] / opt_results['backward_time']:.2f}x")
    
    if opt_results['backward_memory'] > 0.01:
        memory_ratio = orig_results['backward_memory'] / opt_results['backward_memory']
        print(f"Memory reduction: {memory_ratio:.2f}x")
    else:
        print("Memory usage too small to measure accurately")
    
    return {
        'original': orig_results,
        'optimized': opt_results,
        'outputs_match': outputs_match,
        'grads_match': grads_match
    }

if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device ID: {device_id}")
    
    # Initialize NVML if available
    if NVML_AVAILABLE and device == "cuda":
        try:
            pynvml.nvmlInit()
            print(f"NVML initialized successfully")
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"GPU {i}: {name.decode() if isinstance(name, bytes) else name}")
                print(f"  Total memory: {memory.total / (1024**2):.2f} MB")
                print(f"  Free memory: {memory.free / (1024**2):.2f} MB")
        except Exception as e:
            print(f"NVML initialization error: {e}")
    
    # Run tests at different scales
    try:
        # Small scale test
        print("\n=== SMALL SCALE TEST ===")
        compare_implementations(
            batch_size=2,
            num_points=5000,
            feat_dim=64,
            k_eig=128,
            device=device,
            device_id=device_id
        )
        
        # Medium scale test
        print("\n=== MEDIUM SCALE TEST ===")
        compare_implementations(
            batch_size=1,
            num_points=50000,
            feat_dim=64,
            k_eig=128,
            device=device,
            device_id=device_id
        )
            
        # Large scale test (similar to user's target size)
        print("\n=== LARGE SCALE TEST ===")
        compare_implementations(
            batch_size=1,
            num_points=120000,
            feat_dim=64,
            k_eig=128,
            device=device,
            device_id=device_id
        )
    except RuntimeError as e:
        print(f"Test failed: {e}")
    finally:
        # Clean up NVML
        if NVML_AVAILABLE and device == "cuda":
            try:
                pynvml.nvmlShutdown()
                print("NVML shutdown complete")
            except:
                pass