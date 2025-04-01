# memory_profiler.py
import torch
import time
import gc
from functools import wraps
from contextlib import contextmanager

def print_memory_stats(label):
    print(f"\n=== {label} ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def profile_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear cache before profiling
        torch.cuda.empty_cache()
        gc.collect()
        
        # Record initial memory
        print_memory_stats(f"Before {func.__name__}")
        torch.cuda.reset_peak_memory_stats()
        
        # Time execution
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # Record final memory
        print_memory_stats(f"After {func.__name__}")
        print(f"Duration: {duration:.4f} seconds")
        
        return result
    return wrapper

@contextmanager
def memory_efficient_computation():
    """Context manager for memory-intensive operations"""
    try:
        yield
    finally:
        # Force immediate cleanup
        gc.collect()
        torch.cuda.empty_cache()