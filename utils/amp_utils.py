from contextlib import contextmanager
import torch

@contextmanager
def disable_amp():
    """Context manager to temporarily disable AMP"""
    prev_state = torch.is_autocast_enabled()
    torch.set_autocast_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_enabled(prev_state)