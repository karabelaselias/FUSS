import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import time
import numpy as np

# Import all similarity implementations
from network_module import OptimizedSparseSimilarity
from sparse_similarity_optim import HighPerformanceSparseSimilarity
from improved_similarity import ImprovedSparseSimilarity  # New implementation
from a100_similarity import A100OptimizedSparseSimilarity

# Original reference implementation for validation
class Similarity(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard
    
    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))
        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret


def compare_implementations(args, device='cuda:3'):
    """
    Compare results between original Similarity and the optimized sparse versions
    
    Args:
        args: Command line arguments
        device: CUDA device to use
        
    Returns:
        Dict containing error metrics
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    # Small input size for validation
    small_n = min(args.num_points, 1000)  # Use 1000 points max for comparison
    
    # Create input data
    x = torch.randn(1, small_n, args.feat_dim, device=device)
    y = torch.randn(1, small_n, args.feat_dim, device=device)
    
    # Normalize features
    feat_x = F.normalize(x, dim=-1, p=2)
    feat_y = F.normalize(y, dim=-1, p=2)
    
    # Create similarity matrix
    with torch.no_grad():
        similarity = torch.matmul(feat_x, feat_y.transpose(-1, -2))
    
    # 1. Compute result with original Similarity
    original_sim = Similarity(tau=args.tau, hard=args.hard).to(device)
    with torch.no_grad():
        original_result = original_sim(similarity)
    
    # 2. Compute result with OptimizedSparseSimilarity
    optimized_sparse = OptimizedSparseSimilarity(
        tau=args.tau,
        k_neighbors=args.k_neighbors,
        chunk_size=args.chunk_size,
        hard=args.hard
    ).to(device)
    
    with torch.no_grad():
        optimized_sparse_result = optimized_sparse(similarity)
        # Convert sparse to dense for comparison
        optimized_dense = optimized_sparse_result.to_dense()
    
    # 3. Compute result with HighPerformanceSparseSimilarity
    high_perf_sparse = A100OptimizedSparseSimilarity(
        tau=args.tau,
        k_neighbors=args.k_neighbors,
        chunk_size=args.chunk_size,
        hard=args.hard
    ).to(device)
    
    with torch.no_grad():
        high_perf_sparse_result = high_perf_sparse(similarity)
        # Convert sparse to dense for comparison
        high_perf_dense = high_perf_sparse_result.to_dense()
        
    # 4. Compute result with ImprovedSparseSimilarity
    improved_sparse = ImprovedSparseSimilarity(
        tau=args.tau,
        k_neighbors=args.k_neighbors,
        chunk_size=args.chunk_size,
        streams=args.streams,
        hard=args.hard
    ).to(device)
    
    with torch.no_grad():
        improved_sparse_result = improved_sparse(similarity)
        # Convert sparse to dense for comparison
        improved_dense = improved_sparse_result.to_dense()
    
    # Calculate errors
    optimized_error = (original_result - optimized_dense).abs().mean().item()
    optimized_rel_error = optimized_error / original_result.abs().mean().item()
    
    high_perf_error = (original_result - high_perf_dense).abs().mean().item()
    high_perf_rel_error = high_perf_error / original_result.abs().mean().item()
    
    improved_error = (original_result - improved_dense).abs().mean().item()
    improved_rel_error = improved_error / original_result.abs().mean().item()
    
    # Print results
    print("\nValidation Results (n={})".format(small_n))
    print("-" * 50)
    print(f"Original vs OptimizedSparse:")
    print(f"  Absolute Error: {optimized_error:.6f}")
    print(f"  Relative Error: {optimized_rel_error:.6f}")
    print(f"Original vs HighPerformanceSparse:")
    print(f"  Absolute Error: {high_perf_error:.6f}")
    print(f"  Relative Error: {high_perf_rel_error:.6f}")
    print(f"Original vs ImprovedSparse:")
    print(f"  Absolute Error: {improved_error:.6f}")
    print(f"  Relative Error: {improved_rel_error:.6f}")
    print("-" * 50)
    
    # Explain the error
    print("\nNote on approximation error:")
    print("The relative error is expected when using a sparse approximation.")
    print("The original implementation uses a full softmax across all points,")
    print("while the sparse implementations only consider the top-k values.")
    print("This tradeoff enables scaling to much larger point clouds.")
    print("-" * 50)
    
    return {
        "optimized_error": optimized_error,
        "optimized_rel_error": optimized_rel_error,
        "high_perf_error": high_perf_error,
        "high_perf_rel_error": high_perf_rel_error,
        "improved_error": improved_error,
        "improved_rel_error": improved_rel_error
    }


def minimal_working_example(args, implementation='original'):
    """
    A minimal working example demonstrating memory-efficient similarity computation
    
    Args:
        args: Command line arguments
        implementation: Which similarity implementation to use
            - 'original': Original OptimizedSparseSimilarity
            - 'high_perf': HighPerformanceSparseSimilarity
            - 'improved': ImprovedSparseSimilarity (optimized for backward pass)
        
    Returns:
        Dict containing memory usage statistics
    """
    
    # Define a simple feature extractor
    class SimpleFeatureExtractor(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(SimpleFeatureExtractor, self).__init__()
            self.linear = nn.Linear(in_dim, out_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    class SimpleNetwork(nn.Module):
        def __init__(self, in_dim, hidden_dim, args, implementation):
            super().__init__()
            self.feature_extractor = nn.Linear(in_dim, hidden_dim)
            
            if implementation == 'original':
                self.similarity = OptimizedSparseSimilarity(
                    tau=args.tau,
                    k_neighbors=args.k_neighbors,
                    chunk_size=args.chunk_size,
                    hard=args.hard
                )
            elif implementation == 'high_perf':
                self.similarity = HighPerformanceSparseSimilarity(
                    tau=args.tau,
                    k_neighbors=args.k_neighbors,
                    chunk_size=args.chunk_size,
                    streams=args.streams,
                    hard=args.hard
                )
            elif implementation == 'improved':
                self.similarity = ImprovedSparseSimilarity(
                    tau=args.tau,
                    k_neighbors=args.k_neighbors,
                    chunk_size=args.chunk_size,
                    streams=args.streams,
                    hard=args.hard
                )
            elif implementation == 'a100':
                self.similarity = A100OptimizedSparseSimilarity(
                    tau=args.tau,
                    k_neighbors=args.k_neighbors,
                    chunk_size=args.chunk_size,
                    streams=args.streams,
                    hard=args.hard
                )
            else:
                raise ValueError(f"Unknown implementation: {implementation}")
        
        def forward(self, x, y):
            feat_x = self.feature_extractor(x)
            feat_y = self.feature_extractor(y)
            return self.similarity(feat_x, feat_y)
            
    def track_memory():
        """Accurately track CUDA memory usage in GB"""
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return {
            "allocated": memory_allocated,
            "reserved": memory_reserved,
            "max_allocated": max_memory
        }
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Running with {args.num_points} points, k={args.k_neighbors}, chunk={args.chunk_size}, implementation={implementation}")
    
    try:
        batch_size = 1
        # Create input data
        x = torch.randn(1, args.num_points, args.feat_dim, device='cuda:3')
        y = torch.randn(1, args.num_points, args.feat_dim, device='cuda:3')
        
        # Create model
        model = SimpleNetwork(
            args.feat_dim,
            args.feat_dim,
            args,
            implementation
        ).to(device='cuda:3')
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler()
        
        # Forward pass
        start_time = time.time()
        optimizer.zero_grad()
        
        # Forward pass with AMP
        # Reset memory stats before forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            perm_matrix = model(x, y)
        torch.cuda.synchronize()  # Ensure completion before measuring
        forward_time = time.time() - start_time
        forward_memory = track_memory()
        
        # Print statistics
        print(f"Forward pass: {forward_time:.4f}s")
        print(f"Memory: {forward_memory['max_allocated']:.2f}GB")
        print(f"Sparse values: {perm_matrix.values().shape}")
        print(f"Sparsity ratio: {perm_matrix.values().shape[0] / (batch_size * args.num_points * args.num_points):.8f}")
        
        # Simple backward pass
        # Reset memory stats before backward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = perm_matrix.values().sum()
        
        start_time = time.time()
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()  # Ensure completion before measuring
        backward_time = time.time() - start_time
        backward_memory = track_memory()
        
        # Print with more precision
        print(f"Forward memory: {forward_memory['allocated']:.6f} GB (allocated), "
          f"{forward_memory['max_allocated']:.6f} GB (peak)")
        print(f"Backward memory: {backward_memory['allocated']:.6f} GB (allocated), "
          f"{backward_memory['max_allocated']:.6f} GB (peak)")
        
        # Clean up
        del x, y, perm_matrix, model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "points": args.num_points,
            "k_neighbors": args.k_neighbors,
            "chunk_size": args.chunk_size,
            "implementation": implementation,
            "forward_time": forward_time,
            "forward_memory": forward_memory["max_allocated"],
            "backward_time": backward_time,
            "backward_memory": backward_memory["max_allocated"],
            "success": True
        }
    
    except Exception as e:
        print(f"Error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "points": args.num_points,
            "k_neighbors": args.k_neighbors,
            "chunk_size": args.chunk_size,
            "implementation": implementation,
            "error": str(e),
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(description='Optimized Sparse Similarity Benchmark')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points (N)')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Top-k neighbors')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Processing chunk size')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--tau', type=float, default=0.07, help='Temperature parameter')
    parser.add_argument('--hard', action='store_true', help='Use hard assignment')
    parser.add_argument('--streams', type=int, default=4, help='Number of CUDA streams')
    parser.add_argument('--validate', action='store_true', help='Run validation comparison')
    parser.add_argument('--compare', action='store_true', help='Compare all implementations')
    parser.add_argument('--implementation', type=str, default='improved', 
                        choices=['original', 'high_perf', 'improved', 'a100'], 
                        help='Which implementation to use')
    args = parser.parse_args()
    
    # Run validation if requested
    if args.validate:
        compare_implementations(args)
    
    # Run benchmark for all implementations if requested
    if args.compare:
        print("\n=== Running benchmarks for all implementations ===\n")
        original_result = minimal_working_example(args, implementation='original')
        high_perf_result = minimal_working_example(args, implementation='a100')
        improved_result = minimal_working_example(args, implementation='improved')
        
        # Print comparison
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(f"                    | {'Original':^12} | {'A100':^15} | {'Improved':^12} | {'Best Speedup':^10}")
        print("-" * 80)
        print(f"Forward Time (s)    | {original_result['forward_time']:^12.4f} | {high_perf_result['forward_time']:^15.4f} | {improved_result['forward_time']:^12.4f} | {original_result['forward_time']/min(high_perf_result['forward_time'], improved_result['forward_time']):^10.2f}x")
        print(f"Backward Time (s)   | {original_result['backward_time']:^12.4f} | {high_perf_result['backward_time']:^15.4f} | {improved_result['backward_time']:^12.4f} | {original_result['backward_time']/min(high_perf_result['backward_time'], improved_result['backward_time']):^10.2f}x")
        print(f"Total Time (s)      | {original_result['forward_time']+original_result['backward_time']:^12.4f} | {high_perf_result['forward_time']+high_perf_result['backward_time']:^15.4f} | {improved_result['forward_time']+improved_result['backward_time']:^12.4f} | {(original_result['forward_time']+original_result['backward_time'])/min(high_perf_result['forward_time']+high_perf_result['backward_time'], improved_result['forward_time']+improved_result['backward_time']):^10.2f}x")
        print("-" * 80)
        
        # Recommend best implementation
        forward_times = {
            'original': original_result['forward_time'],
            'a100': high_perf_result['forward_time'],
            'improved': improved_result['forward_time']
        }
        backward_times = {
            'original': original_result['backward_time'],
            'a100': high_perf_result['backward_time'],
            'improved': improved_result['backward_time']
        }
        total_times = {
            'original': original_result['forward_time'] + original_result['backward_time'],
            'a100': high_perf_result['forward_time'] + high_perf_result['backward_time'],
            'improved': improved_result['forward_time'] + improved_result['backward_time']
        }
        
        best_forward = min(forward_times, key=forward_times.get)
        best_backward = min(backward_times, key=backward_times.get)
        best_total = min(total_times, key=total_times.get)
        
        print(f"\nBest Forward Pass: {best_forward} ({forward_times[best_forward]:.4f}s)")
        print(f"Best Backward Pass: {best_backward} ({backward_times[best_backward]:.4f}s)")
        print(f"Best Overall: {best_total} ({total_times[best_total]:.4f}s)")
        
        return
    
    # Default: run single implementation benchmark
    print(f"Testing N={args.num_points}, k={args.k_neighbors}, chunk={args.chunk_size}")
    result = minimal_working_example(args, implementation=args.implementation)


if __name__ == "__main__":
    main()