"""
Optimizer registry and wrappers for ProNDF.

Currently only contains the Adam optimizer, but allows for easy extension to 
other/custom optimizers if needed.
"""

import torch


# Optimizer registry for storing different types of optimizers
OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    """
    Decorator to register a optimizer type with the given name.
    
    Args:
        name (str): The name of the optimizer type to register.
    
    Returns:
        function: The decorator function that registers the optimizer.
    """
    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator


@register_optimizer("Adam")
class Adam(torch.optim.Adam):
    """Adam optimizer wrapper for registry usage."""
    pass
