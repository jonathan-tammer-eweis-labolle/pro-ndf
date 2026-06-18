"""
blocks.py
This module contains model block classes and a registry to store them.
Additional or custom blocks can be registered by the user if desired, either in this file or inline.
Registries are used to enable serialization by PyTorch Lightning's checkpointing 
system and automatic hyperparameter saving, as class objects can not be serialized.
Example usage:
    # Importing and using the registry and adding a custom block
    from prondf.blocks import BLOCK_REGISTRY, register_block

    # Registering a custom block (inline or in this file)
    @register_block("CustomBlock")
    class CustomBlock(Base_Block):
        def __init__(self, d_in, d_out):
            super(CustomBlock, self).__init__()
            self.fc = nn.Linear(d_in, d_out)
        
        def forward(self, x):
            return self.fc(x)
To view registered blocks, use:
    print(BLOCK_REGISTRY.keys())
"""

import torch
from torch import nn
from .act_fns import ACT_FN_REGISTRY
from .utils import reparameterization_trick


# Block registry for storing different types of blocks
BLOCK_REGISTRY = {}

def register_block(name):
    """
    Decorator to register a block type with the given name.
    
    Args:
        name (str): The name of the block type to register.
    
    Returns:
        function: The decorator function that registers the block.
    """
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator


# Model blocks
class Base_Block(nn.Module):
    """
    Base class for all blocks. Should not be instantiated directly.
    """
    def __init__(
            self,
            probabilistic_output: bool = False,
            ):
        """
        Initializes base block and stores probabilistic output flag.
        """
        super(Base_Block, self).__init__()
        if type(self) is Base_Block:
            raise NotImplementedError("Base_Block should not be instantiated directly.")
        self.probabilistic_output = probabilistic_output
    
    def forward(self, x):
        """
        Runs through network and returns output predition.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    
    def predict_distribution(self, x):
        """
        Provides output prediction distribution. Only used if model is probabilistic.
        """
        raise NotImplementedError("Predict distribution method should be implemented in subclasses.")
        

@register_block("Det_Block")
class Det_Block(Base_Block):
    """
    A deterministic feed-forward neural network block.

    Consists of an arbitrary number of hidden layers with identical activation 
    functions save for the final output layer, whose activation function is specified by 
    the user. Takes in a tensor of shape (batch_size, d_in) and outputs a single 
    deterministic tensor of size d_out. Can be used for both regression and 
    classification tasks depending on specified output activation.

    Inherits from:
        Base_Block
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_layers: list[int],
        hidden_act_fn: str = "Tanh",
        output_act_fn: str = "Identity",
    ):
        """
        Initializes the Det_Block with the specified parameters.
        Args:
            d_in (int): Dimension of the input.
            d_out (int): Dimension of the output.
            hidden_layers (list[int]): List of integers specifying the number of neurons 
                in each hidden layer.
            hidden_act_fn (str): Activation function for hidden layers, selected from 
                ACT_FN_REGISTRY. Defaults to "Tanh" (hyperbolic tangent).
            output_act_fn (str): Activation function for the output layer, selected 
                from ACT_FN_REGISTRY. Defaults to "Identity".
        Raises:
            KeyError: If the specified activation functions are not found in 
                ACT_FN_REGISTRY.
        """
        super(Det_Block, self).__init__()
        # Store params
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = hidden_layers
        self.hidden_act_fn = ACT_FN_REGISTRY[hidden_act_fn]()
        self.output_act_fn = ACT_FN_REGISTRY[output_act_fn]()
        # Build architecture
        Block = []
        neurons_in = d_in
        for neurons in hidden_layers:
            Block.append(nn.Linear(neurons_in, neurons))
            Block.append(self.hidden_act_fn)
            neurons_in = neurons
        Block.append(nn.Linear(neurons_in, d_out))  # Deterministic output
        Block.append(self.output_act_fn)
        architecture = nn.Sequential(*Block)
        for layer in architecture:  # Initialize via Xavier uniform
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)  # Initialize biases to zero
        self.architecture = architecture

    def forward(self, x):
        """
        Runs through network and returns output.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in).
        """
        temp = self.architecture(x)
        return temp


@register_block("Prob_Block")
class Prob_Block(Base_Block):
    """
    A probabilistic feed-forward neural network block.

    Consists of an arbitrary number of hidden layers with identical activation 
    functions save for the final output layer, whose activation function is specified by 
    the user. Takes in a tensor of shape (batch_size, d_in) and outputs a tensor of size 
    2*d_out representing parameters of a Gaussian distribution (mean and variance). 
    Should be used only for regression tasks.

    Inherits from:
        Base_Block
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_layers: list[int],
        hidden_act_fn: str = "Tanh",
        output_act_fn: str = "Identity",
    ):
        """
        Initializes the Prob_Block with the specified parameters.
        Args:
            d_in (int): Dimension of the input.
            d_out (int): Dimension of the output.
            hidden_layers (list[int]): List of integers specifying the number of neurons 
                in each hidden layer.
            hidden_act_fn (str): Activation function for hidden layers, selected from 
                ACT_FN_REGISTRY. Defaults to "Tanh" (hyperbolic tangent).
            output_act_fn (str): Activation function for the output layer, selected 
                from ACT_FN_REGISTRY. Defaults to "Identity".
        Raises:
            KeyError: If the specified activation functions are not found in 
                ACT_FN_REGISTRY.
        """
        super(Prob_Block, self).__init__()
        # Store params
        self.probabilistic_output = True
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_layers = hidden_layers
        self.hidden_act_fn = ACT_FN_REGISTRY[hidden_act_fn]()
        self.output_act_fn = ACT_FN_REGISTRY[output_act_fn]()
        # Build architecture
        Block = []
        neurons_in = d_in
        for neurons in hidden_layers:
            Block.append(nn.Linear(neurons_in, neurons))
            Block.append(self.hidden_act_fn)
            neurons_in = neurons
        Block.append(nn.Linear(neurons_in, 2 * d_out))  # Probabilistic output
        Block.append(self.output_act_fn)
        architecture = nn.Sequential(*Block)
        for layer in architecture:  # Initialize via Xavier uniform
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)  # Initialize biases to zero
        self.architecture = architecture

    def forward(self, x):
        """
        Runs through network and samples from output distribution with 
        reparameterization trick.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in).
        """
        temp = self.architecture(x)
        mu, log_var = torch.chunk(temp, 2, dim=-1)
        return reparameterization_trick(mu, log_var)
    
    def distribution_params(self, x):
        """
        Runs through network and provides parameters of the output distribution.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in).
        Returns:
            tuple: A tuple containing the mean and log variance of the output distribution.
        """
        temp = self.architecture(x)
        mu, log_var = torch.chunk(temp, 2, dim=-1)
        return mu, log_var

    def predict_distribution(self, x):
        """
        Runs through network and provides predicted output distribution.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_in).
        """
        mu, log_var = self.distribution_params(x)
        sigma = torch.exp(0.5 * log_var)
        return torch.distributions.normal.Normal(mu, sigma)
    
