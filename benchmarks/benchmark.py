import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
import gc
import time
import numpy as np
from memory_profiler import profile
from network_module import OptimizedSparseSimilarity


def minimal_working_example(args):
    """
    A minimal working example demonstrating memory-efficient similarity computation
    
    Args:
        batch_size: Batch size
        num_points: Number of points (vertices)
        feat_dim: Feature dimension
        k_neighbors: Number of neighbors to keep
        chunk_size: Size of chunks for processing
        force_row_by_row: Whether to process row by row
        
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
        def __init__(self, in_dim, hidden_dim, args):
            super().__init__()
            self.feature_extractor = nn.Linear(in_dim, hidden_dim)
            self.similarity = OptimizedSparseSimilarity(
                tau=0.07,
                k_neighbors=args.k_neighbors,
                chunk_size=args.chunk_size,
                hard=False
            )
        
        def forward(self, x, y):
            feat_x = self.feature_extractor(x)
            feat_y = self.feature_extractor(y)
            return self.similarity(feat_x, feat_y)
            
    # Track memory usage
    def track_memory():
        torch.cuda.synchronize()
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
    
    #print(f"Running with {num_points} points, k={k_neighbors}, chunk={chunk_size}, row_by_row={force_row_by_row}")
    
    try:
        batch_size = 1
        # Create input data
        x = torch.randn(1, args.num_points, args.feat_dim, device='cuda:3')
        y = torch.randn(1, args.num_points, args.feat_dim, device='cuda:3')
        
        # Create model
        model = SimpleNetwork(
            args.feat_dim,
            args.feat_dim,
            args
        ).to(device='cuda:3')
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler()
        
        
        # Forward pass
        start_time = time.time()
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            perm_matrix = model(x, y)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        forward_memory = track_memory()
        
        # Print statistics
        print(f"Forward pass: {forward_time:.4f}s")
        print(f"Memory: {forward_memory['max_allocated']:.2f}GB")
        print(f"Sparse values: {perm_matrix.values().shape}")
        print(f"Sparsity ratio: {perm_matrix.values().shape[0] / (batch_size * args.num_points * args.num_points):.8f}")
        
        # Simple backward pass
        torch.cuda.reset_peak_memory_stats()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = perm_matrix.values().sum()
        
        start_time = time.time()
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #loss.backward()
        #optimizer.step()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time
        backward_memory = track_memory()
        
        print(f"Backward pass: {backward_time:.4f}s")
        print(f"Memory: {backward_memory['max_allocated']:.2f}GB")
        
        # Clean up
        del x, y, perm_matrix, model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "points": args.num_points,
            "k_neighbors": args.k_neighbors,
            "chunk_size": args.chunk_size,
            #"force_row_by_row": force_row_by_row,
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
            #"force_row_by_row": force_row_by_row,
            "error": str(e),
            "success": False
        }

def optimize_hyperparameters(max_points=150000, feat_dim=128):
    """
    Find optimal hyperparameter settings for the MemoryEfficientSimilarity module
    
    Args:
        max_points: Maximum number of points to test
        feat_dim: Feature dimension
        
    Returns:
        Recommended hyperparameters
    """
    import numpy as np
    import time
    
    # Configurations to test
    configs = [
        # (num_points, k_neighbors, chunk_size, force_row_by_row)
        (10000, 10, 50, True),
        (10000, 10, 50, False),
        (10000, 15, 50, True),
        (10000, 15, 100, True),
        (20000, 10, 50, True),
        (20000, 15, 50, True),
        (20000, 15, 20, True),
        (50000, 10, 20, True),
        (50000, 5, 20, True),
        (100000, 5, 10, True),
        (100000, 3, 10, True)
    ]
    
    print(f"Testing {len(configs)} configurations...")
    print("=" * 80)
    
    results = []
    for config in configs:
        num_points, k_neighbors, chunk_size, force_row_by_row = config
        
        print(f"\nTesting {num_points} points, k={k_neighbors}, chunk={chunk_size}, row-by-row={force_row_by_row}")
        print("-" * 60)
        
        # Run test
        try:
            start_time = time.time()
            result = minimal_working_example(
                batch_size=1,
                num_points=num_points,
                feat_dim=feat_dim,
                k_neighbors=k_neighbors,
                chunk_size=chunk_size,
                force_row_by_row=force_row_by_row
            )
            total_time = time.time() - start_time
            
            if result["success"]:
                result["total_time"] = total_time
                result["total_memory"] = result["forward_memory"] + result["backward_memory"]
                
                print(f"Total time: {total_time:.2f}s")
                print(f"Total memory: {result['total_memory']:.2f}GB")
                
                results.append(result)
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 60)
        except Exception as e:
            print(f"Error testing configuration: {e}")
    
    # Analyze results
    if results:
        # Sort by memory usage
        results.sort(key=lambda x: x["total_memory"])
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print("\nTop 3 most memory-efficient configurations:")
        for i, r in enumerate(results[:3]):
            print(f"{i+1}. Points: {r['points']}, k: {r['k_neighbors']}, chunk: {r['chunk_size']}, "
                  f"row-by-row: {r['force_row_by_row']}")
            print(f"   Memory: {r['total_memory']:.2f}GB, Time: {r['total_time']:.2f}s")
        
        # Get best config
        best = results[0]
        
        print("\nRECOMMENDED CONFIGURATION:")
        print(f"k_neighbors = {best['k_neighbors']}")
        print(f"chunk_size = {best['chunk_size']}")
        print(f"force_row_by_row = {best['force_row_by_row']}")
        print(f"Total memory: {best['total_memory']:.2f}GB")
        
        # Estimate memory for max_points
        if len(results) >= 3:
            try:
                # Use points that were successfully processed
                point_sizes = np.array([r["points"] for r in results])
                memory_usage = np.array([r["total_memory"] for r in results])
                
                # Simple quadratic model: memory = a*points^2 + b*points + c
                from scipy.optimize import curve_fit
                
                def memory_model(n, a, b, c):
                    return a * n**2 + b * n + c
                
                popt, _ = curve_fit(memory_model, point_sizes, memory_usage)
                a, b, c = popt
                
                # Predict for max_points
                predicted_memory = memory_model(max_points, a, b, c)
                
                print(f"\nPredicted memory for {max_points} points: {predicted_memory:.2f}GB")
                
                # Recommended settings for max_points
                if predicted_memory > 30:  # If predicted to use more than 30GB
                    print("\nFor maximum memory efficiency with very large point clouds:")
                    print("k_neighbors = 3")
                    print("chunk_size = 10")
                    print("force_row_by_row = True")
                    print("\nNote: Consider further reducing k_neighbors or using vertex sampling")
                else:
                    # Scale k_neighbors and chunk_size inversely with predicted memory
                    k_scale = min(1.0, 30 / predicted_memory)
                    recommended_k = max(3, int(best["k_neighbors"] * k_scale))
                    recommended_chunk = max(10, int(best["chunk_size"] * k_scale))
                    
                    print("\nFor balancing speed and memory with large point clouds:")
                    print(f"k_neighbors = {recommended_k}")
                    print(f"chunk_size = {recommended_chunk}")
                    print(f"force_row_by_row = True")
            except Exception as e:
                print(f"\nCould not fit memory model: {e}")
                
                # Fallback recommendations for large point clouds
                print("\nRecommended settings for very large point clouds:")
                print("k_neighbors = 3-5")
                print("chunk_size = 10-20")
                print("force_row_by_row = True")
        
        # Usage instructions
        print("\n" + "=" * 80)
        print("USAGE INSTRUCTIONS")
        print("=" * 80)
        print("1. Replace the existing Similarity module with MemoryEfficientSimilarity")
        print("2. Configure with the recommended parameters")
        print("3. If you still encounter memory issues:")
        print("   - Further reduce k_neighbors (minimum 3)")
        print("   - Reduce chunk_size")
        print("   - Ensure force_row_by_row=True")
        print("   - Consider downsampling your point clouds")
        
        return results
    else:
        print("No successful configurations found.")
        return []

def run_test():
    """
    Run the minimal working example with recommended settings
    for large point clouds
    """
    print("Testing memory-efficient similarity with recommended settings for large point clouds")
    
    # These settings should work well for large point clouds while minimizing memory usage
    minimal_working_example(
        batch_size=1,
        num_points=10000,  # Adjust this based on your GPU memory
        feat_dim=128,
        k_neighbors=5,      # Reduced number of neighbors
        chunk_size=8192,      # Small chunk size
        force_row_by_row=True  # Maximum memory efficiency
    )
    
    print("\nTo find optimal parameters for your specific GPU:")
    print("optimize_hyperparameters(max_points=150000)")

def main():
    parser = argparse.ArgumentParser(description='Optimized Sparse Similarity Benchmark')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points (N)')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Top-k neighbors')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Processing chunk size')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    args = parser.parse_args()
    
    print(f"Testing N={args.num_points}, k={args.k_neighbors}, chunk={args.chunk_size}")
    result = minimal_working_example(args)

if __name__ == "__main__":
    #run_test()
    main()