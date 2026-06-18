"""
act_fns.py
This module contains activation function classes and a registry to store them.
Additional or custom activation functions can be registered by the user if desired, 
either in this file or inline.
Registries are used to enable serialization by PyTorch Lightning's checkpointing 
system and automatic hyperparameter saving, as class objects can not be serialized.
Example usage:
    # Importing and using the registry and adding a custom activation function
    from act_fns import ACT_FN_REGISTRY, register_act_fn

    # Registering a custom activation function (inline or in this file)
    @register_act_fn("CustomActFn")
    class CustomActFn(nn.Module):
        def forward(self, x):
            return torch.sin(x)  # Example custom activation function
To view registered activation functions, use:
    print(ACT_FN_REGISTRY.keys())
"""

from torch import nn


# Activation function registry for storing different types of activation functions
ACT_FN_REGISTRY = {}

def register_act_fn(name):
    """
    Decorator to register a act_fn type with the given name.
    
    Args:
        name (str): The name of the act_fn type to register.
    
    Returns:
        function: The decorator function that registers the act_fn.
    """
    def decorator(cls):
        ACT_FN_REGISTRY[name] = cls
        return cls
    return decorator


@register_act_fn("Identity")
class Identity(nn.Identity):
    """Identity activation wrapper for registry usage."""
    pass


@register_act_fn("Tanh")
class Tanh(nn.Tanh):
    """Hyperbolic tangent activation wrapper for registry usage."""
    pass


@register_act_fn("ReLU")
class ReLU(nn.ReLU):
    """ReLU activation wrapper for registry usage."""
    pass


@register_act_fn("Sigmoid")
class Sigmoid(nn.Sigmoid):
    """Sigmoid activation wrapper for registry usage."""
    pass


@register_act_fn("LeakyReLU")
class LeakyReLU(nn.LeakyReLU):
    """LeakyReLU activation wrapper for registry usage."""
    pass


@register_act_fn("Softmax")
class Softmax(nn.Softmax):
    """Softmax activation wrapper for registry usage."""
    pass
