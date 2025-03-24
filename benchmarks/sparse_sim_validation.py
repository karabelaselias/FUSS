import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the main component (assuming you've saved the first artifact)
from sparse_similarity_mwe import A100OptimizedSparseSimilarity, CompleteModel


def validate_amp_numerical_stability():
    """
    Validate the numerical stability of the SparseSimilarity component with AMP
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running validation on: {device}")
    
    # Create consistent test data
    torch.manual_seed(42)
    batch_size = 1
    num_points = 128
    input_dim = 256
    embed_dim = 64
    k_neighbors = 10
    
    # Create input data
    x = torch.randn(batch_size, num_points, input_dim, device=device)
    
    # Test configurations
    configs = [
        {"name": "FP32 (no AMP)", "use_amp": False, "hard": False},
        {"name": "FP16 (AMP)", "use_amp": True, "hard": False},
        {"name": "FP32 Hard (no AMP)", "use_amp": False, "hard": True},
        {"name": "FP16 Hard (AMP)", "use_amp": True, "hard": True}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Create model with specific config
        model = CompleteModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            k_neighbors=k_neighbors,
            hard=config["hard"],
            use_half=config["use_amp"]
        )
        model.to(device)
        
        # Create optimizer and scaler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler(enabled=config["use_amp"])
        
        # Storage for training metrics
        losses = []
        grad_norms = []
        timings = []
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            # Start timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            # Forward pass
            with autocast(device_type='cuda', enabled=config["use_amp"]):
                output = model(x)
                loss = F.mse_loss(output, x)  # Reconstruction loss
                
            # Backward pass with scaler
            optimizer.zero_grad()
            if config["use_amp"]:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Calculate gradient norm after unscaling
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                optimizer.step()
            
            # End timing
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            
            # Store metrics
            losses.append(loss.item())
            grad_norms.append(total_norm)
            timings.append(elapsed_time)
            
            print(f"Epoch {epoch}: Loss: {loss.item():.6f}, Grad norm: {total_norm:.6f}, Time: {elapsed_time:.2f}ms")
        
        # Store results
        results[config["name"]] = {
            "losses": losses,
            "grad_norms": grad_norms,
            "timings": timings
        }
    
    # Create plots directory
    os.makedirs("amp_validation_plots", exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["losses"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("amp_validation_plots/loss_comparison.png")
    
    # Plot gradient norms
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["grad_norms"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Comparison Across Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("amp_validation_plots/grad_norm_comparison.png")
    
    # Plot timings
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["timings"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Time (ms)")
    plt.title("Training Time Comparison Across Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("amp_validation_plots/timing_comparison.png")
    
    # Calculate average metrics
    print("\nAverage Metrics:")
    print("-" * 60)
    print(f"{'Configuration':<20} {'Avg Loss':<12} {'Avg Grad Norm':<15} {'Avg Time (ms)':<12}")
    print("-" * 60)
    
    for name, data in results.items():
        avg_loss = sum(data["losses"]) / len(data["losses"])
        avg_grad = sum(data["grad_norms"]) / len(data["grad_norms"])
        avg_time = sum(data["timings"]) / len(data["timings"])
        print(f"{name:<20} {avg_loss:<12.6f} {avg_grad:<15.6f} {avg_time:<12.2f}")

def test_gradient_flow():
    """
    Test gradient flow through the sparse similarity component
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting gradient flow on: {device}")
    
    # Set up test case
    torch.manual_seed(42)
    input_dim = 32
    embed_dim = 16
    num_points = 10
    k_neighbors = 3
    
    # Create a small input tensor that we can easily inspect
    x = torch.randn(1, num_points, input_dim, device=device, requires_grad=True)
    
    # Create the similarity component directly
    sim = A100OptimizedSparseSimilarity(tau=0.1, k_neighbors=k_neighbors, use_half=True)
    
    # Test both with and without autocast
    modes = [
        {"name": "No autocast", "use_autocast": False},
        {"name": "With autocast", "use_autocast": True}
    ]
    
    for mode in modes:
        print(f"\nTesting: {mode['name']}")
        
        # Create encoder to get embeddings
        encoder = nn.Linear(input_dim, embed_dim).to(device)
        
        # Forward pass
        with autocast(device_type='cuda', enabled=mode["use_autocast"]):
            # Get embeddings
            embeddings = encoder(x)
            
            # Process through similarity
            sparse_sim = sim(embeddings)
            
            # Compute some loss (e.g., sum of values)
            loss = sparse_sim.values().sum()
        
        # Backward pass
        loss.backward()
        
        # Check if gradients flowed through
        has_grad_x = x.grad is not None and not torch.isnan(x.grad).any() and not torch.isinf(x.grad).any()
        
        print(f"Input gradients exists and valid: {has_grad_x}")
        if has_grad_x:
            print(f"Input gradient norm: {x.grad.norm().item()}")
            print(f"Input gradient min/max: {x.grad.min().item()}/{x.grad.max().item()}")

def main():
    print("Running AMP validation tests for A100OptimizedSparseSimilarity")
    validate_amp_numerical_stability()
    test_gradient_flow()
    print("\nValidation completed!")

if __name__ == "__main__":
    main()