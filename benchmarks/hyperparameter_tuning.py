import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynvml

from improved_similarity import ImprovedSparseSimilarity
from a100_similarity import A100OptimizedSparseSimilarity
from torch_sparse_mm import sparse_mm


def get_gpu_memory_usage(gpu_id):
    """Get GPU memory usage in GB directly from NVML"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024**3)  # Convert to GB

class SimpleNetwork(nn.Module):
    def __init__(self, args, params):
        super().__init__()
        self.feature_extractor = nn.Linear(args.feat_dim, args.feat_dim)
        self.similarity = A100OptimizedSparseSimilarity(
            tau=params['tau'],
            k_neighbors=params['k_neighbors'],
            chunk_size=params['chunk_size'],
            streams=params['streams'],
            hard=args.hard,
            use_half=False
        )
    
    def forward(self, x, y):
        feat_x = self.feature_extractor(x)
        feat_y = self.feature_extractor(y)
        return self.similarity(feat_x, feat_y)

def tune_hyperparameters(args):
    """
    Hyperparameter tuning for ImprovedSparseSimilarity
    
    Args:
        args: Command line arguments
    
    Returns:
        DataFrame with tuning results and recommended hyperparameters
    """
    # Define hyperparameter search space
    param_grid = {
        'k_neighbors': [3, 5, 10, 15, 20, 40],
        'chunk_size': [2048, 4096, 8192, 16384],
        'tau': [0.07],
        'streams': [1, 2, 4, 8]  # Number of streams (only relevant for some implementations)
    }
    
    # Get device
    device = f'cuda:{args.gpu}'
    
    # Setup test data
    torch.manual_seed(42)  # For reproducibility
    batch_size = 1
    
    # Track results
    results = []
    
    # Generate all parameter combinations
    param_combinations = []
    for k in param_grid['k_neighbors']:
        for c in param_grid['chunk_size']:
            for t in param_grid['tau']:
                # Only test 1 stream value if not doing full grid search
                stream_values = param_grid['streams'] if args.full_grid else [param_grid['streams'][0]]
                for s in stream_values:
                    param_combinations.append({
                        'k_neighbors': k,
                        'chunk_size': c,
                        'tau': t,
                        'streams': s
                    })
    
    # If quick tuning, take a representative subset
    if args.quick:
        # Take a subset that covers different parts of the parameter space
        subset_indices = np.linspace(0, len(param_combinations)-1, 10, dtype=int)
        param_combinations = [param_combinations[i] for i in subset_indices]
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations")
    print("=" * 60)
    
    def evaluate_params(params):
        """Evaluate hyperparameters with robust error handling"""
        # Get GPU ID from device string
        gpu_id = args.gpu
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        try:
            # Baseline memory
            baseline_memory = get_gpu_memory_usage(gpu_id)
            
            # Adjust chunk size if needed
            adjusted_chunk_size = min(params['chunk_size'], args.num_points)
            
            # Create input data with careful size management
            x = torch.randn(batch_size, args.num_points, args.feat_dim, device=device)
            y = torch.randn(batch_size, args.num_points, args.feat_dim, device=device)
            z = torch.randn(batch_size, args.num_points, 2*args.feat_dim, device=device)
            
            # Limit k_neighbors to valid range
            max_neighbors = min(params['k_neighbors'], args.num_points - 1)
            params['k_neighbors'] = max_neighbors
            
            # Create model and optimizer
            model = SimpleNetwork(args, {**params, 'k_neighbors': max_neighbors}).to(device)
            optimizer = optim.AdamW([{'params': model.parameters()}], lr=0.001)
            scaler = torch.amp.GradScaler()
            
            # Normalize features
            with torch.no_grad():
                feat_x = F.normalize(x, dim=-1, p=2)
                feat_y = F.normalize(y, dim=-1, p=2)
            
            # Time forward pass
            torch.cuda.synchronize()
            forward_start = time.time()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                optimizer.zero_grad(set_to_none=True)
                perm_matrix = model(feat_x, feat_y)
            
            torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_memory = get_gpu_memory_usage(gpu_id)
            
            # Time backward pass
            torch.cuda.synchronize()
            backward_start = time.time()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                #print(perm_matrix)
                temp = sparse_mm(perm_matrix, z.squeeze()).unsqueeze(0)
                #temp = torch.sparse.mm(perm_matrix, z.squeeze()).unsqueeze(0)
                #print(temp)
                loss = torch.nn.functional.mse_loss(temp, z)
                #loss = perm_matrix.values().sum()
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            
            # Get peak memory
            peak_memory = get_gpu_memory_usage(gpu_id)
            memory_used = peak_memory - baseline_memory
            
            # Clean up
            del x, y, feat_x, feat_y, model, optimizer, perm_matrix
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_time': forward_time + backward_time,
                'memory': memory_used
            }
        
        except Exception as e:
            print(f"Error in parameter evaluation: {e}")
            # Log detailed error information
            import traceback
            traceback.print_exc()
            
            # Return a placeholder result to continue hyperparameter search
            return {
                'forward_time': float('inf'),
                'backward_time': float('inf'),
                'total_time': float('inf'),
                'memory': float('inf')
            }
    
    # Evaluate all parameter combinations
    try:
        for i, params in enumerate(param_combinations):
            print(f"\nEvaluating combination {i+1}/{len(param_combinations)}:")
            print(f"  k_neighbors={params['k_neighbors']}, chunk_size={params['chunk_size']}, "
                  f"tau={params['tau']}, streams={params['streams']}")
            
            # Run evaluation
            metrics = evaluate_params(params)
            
            # Store results
            result = {**params, **metrics}
            results.append(result)
            
            # Print results
            print(f"  Forward: {metrics['forward_time']:.4f}s, Backward: {metrics['backward_time']:.4f}s, "
                  f"Total: {metrics['total_time']:.4f}s, Memory: {metrics['memory']:.2f}GB")
            
            # Save intermediate results to dataframe
            if (i + 1) % 5 == 0 or (i + 1) == len(param_combinations):
                df = pd.DataFrame(results)
                df.to_csv(f"sparse_similarity_tuning_{args.num_points}pts.csv", index=False)
    
    except KeyboardInterrupt:
        print("\nTuning interrupted. Saving partial results...")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save final results
    df.to_csv(f"sparse_similarity_tuning_{args.num_points}pts.csv", index=False)
    
    # Find optimal parameters
    if not df.empty:
        best_forward = df.loc[df['forward_time'].idxmin()]
        best_backward = df.loc[df['backward_time'].idxmin()]
        best_total = df.loc[df['total_time'].idxmin()]
        best_memory = df.loc[df['memory'].idxmin()]
        
        # Print optimal parameters
        print("\n" + "=" * 60)
        print("Optimal Hyperparameters:")
        print("=" * 60)
        print(f"Best Forward Time ({best_forward['forward_time']:.4f}s):")
        print(f"  k_neighbors={int(best_forward['k_neighbors'])}, chunk_size={int(best_forward['chunk_size'])}, "
              f"tau={best_forward['tau']}, streams={int(best_forward['streams'])}")
        
        print(f"\nBest Backward Time ({best_backward['backward_time']:.4f}s):")
        print(f"  k_neighbors={int(best_backward['k_neighbors'])}, chunk_size={int(best_backward['chunk_size'])}, "
              f"tau={best_backward['tau']}, streams={int(best_backward['streams'])}")
        
        print(f"\nBest Total Time ({best_total['total_time']:.4f}s):")
        print(f"  k_neighbors={int(best_total['k_neighbors'])}, chunk_size={int(best_total['chunk_size'])}, "
              f"tau={best_total['tau']}, streams={int(best_total['streams'])}")
        
        print(f"\nBest Memory Usage ({best_memory['memory']:.2f}GB):")
        print(f"  k_neighbors={int(best_memory['k_neighbors'])}, chunk_size={int(best_memory['chunk_size'])}, "
              f"tau={best_memory['tau']}, streams={int(best_memory['streams'])}")
        
        # Create visualizations
        if args.visualize and len(df) > 1:
            # Pivot tables for visualization
            for param in ['k_neighbors', 'chunk_size', 'tau']:
                plt.figure(figsize=(10, 6))
                df_grouped = df.groupby(param).mean().reset_index()
                
                plt.subplot(1, 2, 1)
                plt.plot(df_grouped[param], df_grouped['forward_time'], marker='o', label='Forward')
                plt.plot(df_grouped[param], df_grouped['backward_time'], marker='s', label='Backward')
                plt.plot(df_grouped[param], df_grouped['total_time'], marker='^', label='Total')
                plt.xlabel(param)
                plt.ylabel('Time (s)')
                plt.title(f'Time vs {param}')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(df_grouped[param], df_grouped['memory'], marker='o', color='green')
                plt.xlabel(param)
                plt.ylabel('Memory (GB)')
                plt.title(f'Memory vs {param}')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'tuning_{param}_{args.num_points}pts.png')
                plt.close()
        
        # Return best parameters
        return df, best_total
    
    return df, None

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Sparse Similarity')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points (N)')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--hard', action='store_true', help='Use hard assignment')
    parser.add_argument('--gpu', type=int, default=3, help='GPU ID to use')
    parser.add_argument('--quick', action='store_true', help='Run quick tuning with reduced set')
    parser.add_argument('--full_grid', action='store_true', help='Run full grid search including streams')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    print(f"Running hyperparameter tuning for {args.num_points} points")
    results_df, best_params = tune_hyperparameters(args)
    
    # Output best setting for use in actual benchmark
    if best_params is not None:
        print("\nRecommended command line for benchmark:")
        cmd = (f"python benchmark.py --num_points {args.num_points} "
               f"--k_neighbors {int(best_params['k_neighbors'])} "
               f"--chunk_size {int(best_params['chunk_size'])} "
               f"--tau {best_params['tau']} "
               f"--streams {int(best_params['streams'])}")
        
        if args.hard:
            cmd += " --hard"
        
        print(cmd)

if __name__ == "__main__":
    main()