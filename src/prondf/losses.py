"""
losses.py
This module contains classes for losses, loss-weighting algorithms, and loss 
computations, as well as registries to store them.
Additional or custom losses and algorithms can be registered by the user if desired, 
either in this file or inline.
Registries are used to enable serialization by PyTorch Lightning's checkpointing 
system and automatic hyperparameter saving, as class objects can not be serialized.
Example usage:
    # Importing and using the registry and adding a custom loss
    from prondf.losses import LOSS_REGISTRY, register_loss

    # Registering a custom loss (inline or in this file)
    @register_loss("CustomLoss")
    class CustomLoss(nn.Module):
        def __init__(self):
            super(CustomLoss, self).__init__()
            # Register any buffers or parameters if needed
        def forward(self, mu, y, var, sigma):
            return torch.mean((mu - y) ** 2)  # Example custom loss function
"""

import torch
from torch import nn
from torch.nn import functional as F
import copy

# Basic loss functions
def NLL_loss(mu, var, targets):
    """
    Computes the negative log likelihood loss.
    Args:
        mu (torch.Tensor): Mean of the predicted distribution.
        var (torch.Tensor): Variance of the predicted distribution.
        targets (torch.Tensor): Target values.
    """
    loss = F.gaussian_nll_loss(mu, targets, var, full=True, eps=1e-6) + 6
    return loss


def IS_loss(mu, sigma, targets, alpha=0.05):
    """
    Computes the interval score loss.
    Args:
        preds_dist (torch.distributions.Distribution): Predicted dist. object.
        y (torch.Tensor): Target values.
        alpha (float): Significance level for the interval score, default is 0.05.
        strength (float): Scaling factor for the loss, default is 1.0.
    Returns:
        torch.Tensor: The computed interval score loss.
    """
    mu_lb = mu - 1.96 * sigma
    mu_ub = mu + 1.96 * sigma
    loss = mu_ub - mu_lb
    loss += (targets > mu_ub).float() * 2 / alpha * (targets - mu_ub)
    loss += (targets < mu_lb).float() * 2 / alpha * (mu_lb - targets)
    loss = torch.mean(loss)
    return loss


def KL_div_var_only_loss(var, targets, prior_var=0.01, eps=1e-8):
    """
    Computes the KL divergence loss focusing on variance.
    Args:
        preds_dist (torch.distributions.Distribution): Predicted dist. object.
        y (torch.Tensor): Target values.
        prior_var (float): Prior variance for KL divergence.
        eps (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: The computed KL divergence loss.
    """
    prior_vars = prior_var * torch.ones_like(targets)
    KL_divs = torch.log(torch.sqrt(var) / torch.sqrt(prior_vars) + eps) + prior_vars / (2 * var) - 0.5
    return torch.mean(KL_divs)


# Loss registry for storing different types of loss classes
LOSS_REGISTRY = {}

def register_loss(name):
    """
    Decorator to register a loss type with the given name.
    
    Args:
        name (str): The name of the loss type to register.
    
    Returns:
        function: The decorator function that registers the loss.
    """
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


# Loss context object to store model parameters/outputs to be accessed by loss classes.
# Note: Loss_Context does not inherit from nn.Module to avoid circular references
# when storing the model reference (ProNDF -> loss_handler -> Loss_Context -> model -> ...)
class Loss_Context:
    """
    Context object for losses. Stores model parameters and outputs to be accessed by 
    loss classes.
    User can add more objects to the context if future loss classes require them (for 
    example, adding a probabilistic block's output parameters if they are needed 
    directly for, e.g., a regularizer)
    """
    def __init__(self, model, batch, outputs):
        """
        Initializes the Loss_Context with model parameters and outputs.
        Args:
            model (pl.LightningModule): The model from which to extract parameters or 
                other information if needed.
            batch (dict): The batch of data to be used in loss computation. For typical 
                usage of ProNDF, the batch will be a dict with keys 'source', 'cat', 'num', 
                'targets', where source and cat are one-hot encoded.
            outputs (dict[str, str]): The model outputs to be used in loss 
                computation. Should be containing a dictionary for each block with that 
                block's outputs. For example, if B1 and B3 are probabilistic while B2 is 
                deterministic, outputs would take the following form:
                outputs = {
                    "B1": {
                        "out": torch.Tensor, 
                        "out_dist": torch.distributions.Distribution
                        },
                    "B2": {"out": torch.Tensor},
                    "B3": {
                        "out": torch.Tensor, 
                        "out_dist": torch.distributions.Distribution
                        }
                }
        """
        self.model = model
        self.batch = batch
        self.outputs = outputs


# loss function classes
@register_loss("Base_Loss")
class Base_Loss(nn.Module):
    """
    Base class for all losses. Should not be instantiated directly.
    """
    def __init__(self):
        """
        Initializes the Base_Loss.
        requires_probabilistic_output (bool): If True, assumes the model outputs a 
            distribution object. If False, assumes the model outputs raw tensors.
        """
        super(Base_Loss, self).__init__()
        self.requires_probabilistic_output = False

    def forward(self, context: Loss_Context) -> torch.Tensor:
        """
        Computes the loss. Define in subclasses.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")


@register_loss("Output_MSE_loss")
class Output_MSE_loss(Base_Loss):
    """
    MSE loss on model outputs vs targets. Works with both probabilistic and 
    deterministic outputs.
    """
    def __init__(self):
        """
        Initializes the MSE_loss.
        Args:
            None
        """
        super(Output_MSE_loss, self).__init__()

    def forward(self, context: Loss_Context) -> torch.Tensor:
        """
        Computes the mean squared error loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch['targets']
        # Check if batch is empty (can happen when splitting by source)
        if targets.numel() == 0:
            # Get device from model parameters
            device = next(context.model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if not hasattr(self, "probabilistic_output"):
            self.probabilistic_output = context.model.B3.probabilistic_output
        if self.probabilistic_output:
            # If the model outputs a distribution object, extract the mean
            preds = context.outputs["B3"]["out_dist"]
            preds = preds.mean
        else:
            # If the model outputs raw tensors, use them directly
            preds = context.outputs["B3"]["out"]
        loss = F.mse_loss(preds, targets, reduction="mean")
        return loss


@register_loss("Output_NLL_loss")
class Output_NLL_loss(Base_Loss):
    """
    Negative log likelihood loss on the network output. Network output must be able to 
    output a distribution.
    We add a constant of 6 to the loss to ensure it is positive, as the negative log 
    likelihood can be negative for some distributions. Since a nugget of eps = 1e-6
    is added, this ensures that the loss is always positive and avoids numerical
    instability. See documentation of torch.nn.functional.gaussian_nll_loss:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html
    """

    def __init__(self):
        """
        Initializes the NLL_loss.
        Args:
            None
        """
        super(Output_NLL_loss, self).__init__()
        self.requires_probabilistic_output = True

    def forward(self, context: Loss_Context) -> torch.Tensor:
        """
        Computes the negative log likelihood loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch['targets']
        # Check if batch is empty (can happen when splitting by source)
        if targets.numel() == 0:
            # Get device from model parameters
            device = next(context.model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        preds_dist = context.outputs["B3"]["out_dist"]
        mu = preds_dist.mean
        var = preds_dist.variance
        loss = NLL_loss(mu, var, targets)
        return loss


@register_loss("Output_IS_loss")
class Output_IS_loss(Base_Loss):
    """
    Interval score loss. Assumes 95% CI. Requires network output to be dist. object.
    Typically used as a regularizer with a tunable strength parameter.
    For a definition and in-depth discussion of the interval score, see:
    Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of 
    Multi-fidelity Data Sets by Mora and Eweis-LaBolle et. al. (2023).
    https://arxiv.org/abs/2301.13271
    """

    def __init__(self, alpha=0.05, strength=1.0):
        """
        Initializes the IS_loss with a significance level for the interval score.
        Args:
            alpha (float): Significance level for the interval score, default is 0.05.
            strength (float): Scaling factor for the loss, default is 1.0.
        """
        super(Output_IS_loss, self).__init__()
        self.requires_probabilistic_output = True
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("strength", torch.tensor(strength))

    def forward(self, context: Loss_Context) -> torch.Tensor:
        """
        Computes the interval score loss.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch['targets']
        # Check if batch is empty (can happen when splitting by source)
        if targets.numel() == 0:
            # Get device from model parameters
            device = next(context.model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        preds_dist = context.outputs["B3"]["out_dist"]
        mu = preds_dist.mean
        sigma = preds_dist.stddev
        loss = IS_loss(mu, sigma, targets, alpha = self.alpha.item())
        return loss * self.strength  # Scale by strength factor


@register_loss("Intermediate_KL_Div_Loss")
class Intermediate_KL_Div_Loss(Base_Loss):
    """
    KL divergence loss for variational inference. Requires output of intermediate block 
    to be a distribution object.
    This loss computes the KL divergence between the predicted distribution and a 
    standard normal distribution, focusing only on the variance.
    Should be used to regularize the variance of the outputs of probabillstic 
    intermediate blocks and tuned via strength parameter.
    """

    def __init__(self, block_label = "B1", prior_var = 0.01, eps = 1e-8, strength=1.0):
        """
        Initializes the Intermediate_KL_Div_Loss with a block label, prior variance,
        small epsilon value to avoid division by zero, and a strength factor.
        Args:
            block_label (str): The label of the block whose output to regularize.
            prior_var (float): Prior variance for KL divergence, default is 0.01.
            eps (float): Small value to avoid division by zero, default is 1e-8.
            strength (float): Scaling factor for the loss, default is 1.0.
        """
        super(Intermediate_KL_Div_Loss, self).__init__()
        self.requires_probabilistic_output = True
        self.block_label = block_label
        self.register_buffer("prior_var", torch.tensor(prior_var))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("strength", torch.tensor(strength))

    def forward(self, context: Loss_Context) -> torch.Tensor:
        """
        Computes the KL divergence loss focusing on variance.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            torch.Tensor: The computed loss.
        """
        targets = context.batch['targets']
        # Check if batch is empty (can happen when splitting by source)
        if targets.numel() == 0:
            # Get device from model parameters
            device = next(context.model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        preds_dist = context.outputs[self.block_label]["out_dist"]
        var = preds_dist.variance
        loss = KL_div_var_only_loss(var, targets, prior_var=self.prior_var.item(), eps=self.eps.item())
        return loss * self.strength  # Scale by strength factor

# Data splitting registry
DATA_SPLIT_REGISTRY = {}
def register_data_split(name):
    """
    Decorator to register a data splitting function with the given name.
    
    Args:
        name (str): The name of the data splitting function to register.
    
    Returns:
        function: The decorator function that registers the data splitting function.
    """
    def decorator(cls):
        DATA_SPLIT_REGISTRY[name] = cls
        return cls
    return decorator

# data splitting classes
class Base_Data_Split(nn.Module):
    """
    Base class for data splitting. Should not be instantiated directly.
    """
    def __init__(self):
        super(Base_Data_Split, self).__init__()
        if type(self) is Base_Data_Split:
            raise NotImplementedError(
                "Base_Data_Split should not be instantiated directly."
                )

    def forward(self, context: Loss_Context) -> list[Loss_Context]:
        """
        Splits data into individual components. Define in subclasses.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")
    

@register_data_split("No_Split")
class No_Split(Base_Data_Split):
    """
    Performs no data splitting. Returns the input data as is.
    """
    def __init__(self, config: dict[any, any] = None):
        """
        Initializes the No_Split data split.
        Args:
            config (dict, optional): Config object (not used, but kept for consistency with other data splits).
        """
        super(No_Split, self).__init__()
        self.register_buffer("num_splits", torch.tensor(1))

    def forward(self, context: Loss_Context) -> list[Loss_Context]:
        """
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            out: List containing unmodified context.
        """
        return [context]


@register_data_split("Split_by_Source")
class Split_by_Source(Base_Data_Split):
    """
    Splits data by source. Assumes source is one-hot encoded.
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Split_by_Source with the number of sources.
        Args:
            config (dict): Config object. Should contain 'num_sources' key.
        """
        super(Split_by_Source, self).__init__()
        if "num_sources" not in config:
            raise ValueError(
                "Split_by_Source requires 'num_sources' key in config."
                )
        else:
            self.register_buffer("num_sources", torch.tensor(config["num_sources"]))
            self.register_buffer("num_splits", torch.tensor(config["num_sources"]))
        
    def forward(self, context: Loss_Context) -> list[Loss_Context]:
        """
        Splits data by source. Assumes source is one-hot encoded.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            Out: List of Loss_Context objects with batches and outputs split by source.
        """
        context_splits = []
        source = context.batch['source']
        cat = context.batch['cat']
        num = context.batch['num']
        targets = context.batch['targets']
        outputs = context.outputs
        for ds in range(self.num_sources.item()):
            # Get indices for the current source
            source_mask = source[:, ds] == 1
            # Split batch by source
            # Note: cat and num may be empty tensors (created with empty_like) when qual_in/quant_in are False,
            # but we still split them to maintain consistent batch structure. The model checks these flags
            # before using cat/num, so this is safe.
            source_split = source[source_mask, :]
            cat_split = cat[source_mask, :]
            num_split = num[source_mask, :]
            targets_split = targets[source_mask, :]
            batch_split = {
                'source': source_split,
                'cat': cat_split,
                'num': num_split,
                'targets': targets_split
            }
            # Split outputs by source
            outputs_split = {}
            for block_label, block_outputs in outputs.items():
                block_split = {}
                out_split = block_outputs["out"][source_mask, :]
                if "out_dist" in block_outputs:
                    # Build split dist. object
                    out_dist = block_outputs["out_dist"]
                    mean_split = out_dist.mean[source_mask, :]
                    stddev_split = out_dist.stddev[source_mask, :]
                    out_dist_split = torch.distributions.Normal(mean_split, stddev_split)
                    block_split["out_dist"] = out_dist_split
                # Rebuild outputs dict
                block_split["out"] = out_split
                outputs_split[block_label] = block_split
            # Append new context for the split data
            context_splits.append(Loss_Context(context.model, batch_split, outputs_split))
        return context_splits


@register_data_split("Split_by_Output_Dim")
class Split_by_Output_Dim(Base_Data_Split):
    """
    Splits data by output dimension. Used in multi-output regression tasks where each
    output dimension is treated as a separate task
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Split_by_Output_Dim with the number of output dimensions.
        Args:
            config (dict): Config object. Should contain 'num_outputs' key.
        """
        super(Split_by_Output_Dim, self).__init__()
        if "num_outputs" not in config:
            raise ValueError(
                "Split_by_Output_Dim requires 'num_outputs' key in config."
                )
        else:
            self.register_buffer("num_outputs", torch.tensor(config["num_outputs"]))
            self.register_buffer("num_splits", torch.tensor(config["num_outputs"]))
        
    def forward(self, context: Loss_Context) -> list[Loss_Context]:
        """
        Splits data by source. Assumes source is one-hot encoded.
        Args:
            context (Loss_Context): The context object containing information necessary 
            for calculating the loss.
        Returns:
            Out: List of Loss_Context objects with batches and outputs split by output.
        """
        context_splits = []
        source = context.batch['source']
        cat = context.batch['cat']
        num = context.batch['num']
        targets = context.batch['targets']
        outputs = context.outputs
        for out_idx in range(self.num_outputs.item()):
            #Split targets by output dim
            targets_split = targets[:, out_idx].unsqueeze(1)
            batch_split = {
                'source': source,
                'cat': cat,
                'num': num,
                'targets': targets_split
            }
            # Split outputs by output dim - manually construct to avoid deepcopy issues with computation graph tensors
            outputs_split = {}
            for block_label, block_outputs in outputs.items():
                block_split = {}
                if block_label == "B3":
                    # Split B3 output by output dimension
                    block_split["out"] = block_outputs["out"][:, out_idx].unsqueeze(1)
                    if "out_dist" in block_outputs:
                        # Build split dist. object
                        out_dist = block_outputs["out_dist"]
                        mean_split = out_dist.mean[:, out_idx].unsqueeze(1)
                        stddev_split = out_dist.stddev[:, out_idx].unsqueeze(1)
                        block_split["out_dist"] = torch.distributions.Normal(mean_split, stddev_split)
                else:
                    # For B1 and B2, keep outputs unchanged
                    block_split["out"] = block_outputs["out"]
                    if "out_dist" in block_outputs:
                        block_split["out_dist"] = block_outputs["out_dist"]
                outputs_split[block_label] = block_split
            # Append new context for the split data
            context_splits.append(Loss_Context(context.model, batch_split, outputs_split))
        return context_splits


# Loss weighting algorithm registry
LW_ALG_REGISTRY = {}

def register_lw_alg(name):
    """
    Decorator to register a loss-weighting algorithm with the given name.
    
    Args:
        name (str): The name of the loss-weighting algorithm type to register.
    
    Returns:
        function: The decorator function that registers the loss-weighting algorithm.
    """
    def decorator(cls):
        LW_ALG_REGISTRY[name] = cls
        return cls
    return decorator


# Loss weighting algoithms
class Base_LW_alg(nn.Module):
    """
    Base class for loss weighting algorithms.
    Do not instantiate directly. Subclasses should implement the `forward()` method 
    to compute a weighted sum of loss terms, and optionally `update()` for dynamic 
    schemes.
    """
    def __init__(self):
        super(Base_LW_alg, self).__init__()
        if type(self) is Base_LW_alg:
            raise NotImplementedError(
                "Base_LW_alg should not be instantiated directly."
                )
    
    def forward(self, loss_terms: list[torch.Tensor]) -> torch.Tensor:
        """
        Linear combination of loss terms. Define in subclasses.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
        Returns:
            torch.Tensor: Scalar weighted sum of losses.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")

    def update(self, loss_terms, context: Loss_Context):
        """
        Optionally update the loss weights.
        Override in subclasses when loss weights need to be dynamically updated.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
        """
        pass


@register_lw_alg("No_Weighting")
class No_Weighting(Base_LW_alg):
    """
    This class is used when no loss weighting is desired.
    """
    def __init__(self, **kwargs):
        """
        Initializes the No_Weighting loss weighting algorithm which does not apply any
        weighting to the loss and simply returns it as is.
        Accepts and ignores **kwargs so that loss handlers which inject num_loss_terms
        into all algorithm configs do not cause errors.
        """
        super(No_Weighting, self).__init__()

    def forward(self, loss_terms):
        """
        Applies loss weights to loss terms and sums them.
        Args:
            losses (torch.Tensor): Flattened tensor of losses. Should be scalar or 1D 
            if using no weighting.
        Returns:
            torch.Tensor: Sum of losses.
        """
        return torch.sum(loss_terms)


@register_lw_alg("Two_Moment_Weighting")
class Two_Moment_Weighting(Base_LW_alg):
    """
    Loss weighting algorithm that uses two moment estimates to weight losses.
    This algorithm computes the first and second moments of the gradients of each loss
    term with respect to the model parameters, and uses these moments to compute
    adaptive weights for each loss term. The reference loss term is used to normalize
    the weights of the other loss terms.
    The algorithm is inspired by the papers "Multi-Objective Loss Balancing for 
    Physics-Informed Deep Learning" by Rafael Bischof and Michael Kraus (2021) and 
    "Adam: A method for stochastic optimization" by Diederik P Kingm and Jimmy Ba 
    (2014).
    """

    def __init__(self, num_loss_terms: int, ref_idx: int = 0, alpha1: float = 0.9, alpha2: float = 0.999, eps: float = 1e-8):
        """
        Initializes the Two_Moment_Weighting loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            ref_idx (int): Index of the reference loss term.
            alpha1 (float): Exponential decay rate for first moment.
            alpha2 (float): Exponential decay rate for second moment.
            eps (float): Small value to avoid division by zero.
        """
        super(Two_Moment_Weighting, self).__init__()
        shape = (num_loss_terms,)
        self.register_buffer("lambdas", torch.zeros(shape))
        self.register_buffer("gammas", torch.zeros(shape))
        self.register_buffer("weights", torch.zeros(shape))
        self.register_buffer("alpha1", torch.tensor(alpha1))
        self.register_buffer("alpha2", torch.tensor(alpha2))
        self.register_buffer("eps", torch.tensor(eps))
        self.ref_idx = ref_idx

    def forward(self, loss_terms):
        """
        Applies loss weights to loss terms and sums them.
        Args:
            losses (torch.Tensor): Flattened tensor of losses. Should be scalar or 1D
        """
        # Ensure weights are on the same device as loss_terms
        weights = self.weights.to(loss_terms.device)
        return torch.sum(loss_terms * weights)

    def update(self, loss_terms: list[torch.Tensor], context: Loss_Context):
        """
        Updates the loss weights.
        Args:
            losses (list[torch.Tensor]): List of loss tensors to be weighted.
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
        """
        parameters = list(context.model.parameters())
        step = context.model.global_step if hasattr(context.model, 'global_step') else 0
        # Obtain the reference loss gradients, etc.
        ref_loss = loss_terms[self.ref_idx]
        ref_grads = torch.autograd.grad(
            ref_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        ref_grads_flat = torch.cat([g.view(-1) for g in ref_grads if g is not None])
        ref_grads_max = torch.max(torch.abs(ref_grads_flat))
        ref_grads_max_sq = torch.max(torch.abs(ref_grads_flat) ** 2)
        # Update each loss weight
        for idx, loss in enumerate(loss_terms):
            if idx == self.ref_idx:
                self.weights[idx] = torch.tensor(1.0, device=self.weights.device)
            else:
                # Calculate gradients
                grads = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                grads_flat = torch.cat([g.view(-1) for g in grads if g is not None])
                grads_mean = torch.mean(torch.abs(grads_flat))
                grads_mean_sq = torch.mean(torch.abs(grads_flat) ** 2)
                # Calculate moment estimates w/ nugget to avoid instability
                lambda_hat = ref_grads_max / (grads_mean + self.eps)
                gamma_hat = ref_grads_max_sq / (grads_mean_sq + self.eps)
                # Calculate moving averages
                lambda_mavg = (1 - self.alpha1) * self.lambdas[
                    self.ref_idx
                ] + self.alpha1 * lambda_hat
                gamma_mavg = (1 - self.alpha2) * self.gammas[
                    self.ref_idx
                ] + self.alpha2 * gamma_hat
                # Bias correction
                m = lambda_mavg / (1 - torch.pow(1 - self.alpha1, step + 1))
                v = gamma_mavg / (1 - torch.pow(1 - self.alpha2, step + 1))
                # Calculate weight and update
                self.weights[idx] = m / (torch.sqrt(v) + self.eps)
                self.lambdas[idx] = lambda_mavg
                self.gammas[idx] = gamma_mavg


# TODO: Add gradnorm as a class here
@register_lw_alg("GradNorm")
class GradNorm(Base_LW_alg):
    """TODO: IMPLEMENT. UPDATE DOCSTRING AT LATER DATE."""
    def __init__(self, num_loss_terms: int, ref_idx=0, alpha1=0.9, alpha2=0.999, eps=1e-8):
        """
        Initializes the GradNorm loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            ref_idx (int): Index of the reference loss term.
            alpha1 (float): Exponential decay rate for first moment.
            alpha2 (float): Exponential decay rate for second moment.
            eps (float): Small value to avoid division by zero.
        """
        super(GradNorm, self).__init__()
        raise NotImplementedError(
            "GradNorm is not yet implemented. Please implement the forward and update methods."
        )
        shape = (num_loss_terms,)
        self.register_buffer("lambdas", torch.zeros(shape))
        self.register_buffer("gammas", torch.zeros(shape))
        self.register_buffer("weights", torch.zeros(shape))
        self.register_buffer("alpha1", torch.tensor(alpha1))
        self.register_buffer("alpha2", torch.tensor(alpha2))
        self.register_buffer("eps", torch.tensor(eps))
        self.ref_idx = ref_idx


@register_lw_alg("Fixed_Weights")
class Fixed_Weights(Base_LW_alg):
    """
    Fixed weights loss weighting algorithm. Applies fixed weights to each loss term.
    """
    def __init__(self, num_loss_terms: int, weights: list = None):
        """
        Initializes the Fixed_Weights loss weighting algorithm.
        Args:
            num_loss_terms (int): Number of loss terms to weight.
            weights (list, optional): List of fixed weights for each loss term. If None,
                defaults to equal weights for all terms.
        """
        super(Fixed_Weights, self).__init__()
        shape = (num_loss_terms,)
        if weights is not None:
            if len(weights) != num_loss_terms:
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match number of loss terms ({num_loss_terms})."
                )
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.ones(shape))

    def forward(self, loss_terms):
        """
        Applies loss weights to loss terms and sums them.
        """
        # Ensure weights are on the same device as loss_terms
        weights = self.weights.to(loss_terms.device)
        return torch.sum(loss_terms * weights)


# Loss computation registry
LOSS_HANDLER_REGISTRY = {}

def register_loss_handler(name):
    """
    Decorator to register a loss computation with the given name.
    
    Args:
        name (str): The name of the loss computation type to register.
    
    Returns:
        function: The decorator function that registers the loss computation.
    """
    def decorator(cls):
        LOSS_HANDLER_REGISTRY[name] = cls
        return cls
    return decorator

@register_loss_handler("Base_Loss_Handler")
class Base_Loss_Handler(nn.Module):
    """
    Base class for loss computation handlers. Should not be instantiated directly.
    Subclasses should implement the `forward()` method to compute the loss.
    """
    def __init__(self):
        super(Base_Loss_Handler, self).__init__()
        if type(self) is Base_Loss_Handler:
            raise NotImplementedError(
                "Base_Loss_Handler should not be instantiated directly."
                )

    def build_loss_context(self, model, batch, outputs):
        """
        Builds a Loss_Context object from the model, batch, and outputs.
        Args:
            model (pl.LightningModule): The model from which to extract parameters or 
                other information if needed.
            batch (dict): The batch of data to be used in loss computation. For 
                typical usage of ProNDF, the batch will be a dict with keys 'source', 'cat', 
                'num', 'targets', where source and cat are one-hot encoded.
            outputs (dict[str, str]): The model outputs to be used in loss 
                computation. Should be containing a dictionary for each block with that 
                block's outputs.
        """
        self.context = Loss_Context(model, batch, outputs)


@register_loss_handler("One_Stage_Loss_Handler")
class One_Stage_Loss_Handler(Base_Loss_Handler):
    """
    Loss handler that computes loss in a single stage, splitting data only once.
    Assumes that each provided loss function as well as each data split corresponds to a 
    separate task with respect to loss weighting.
    """
    def __init__(
            self,
            config: dict[any, any],
            ):
        """
        Initializes the One_Stage_Loss_Handler with a configuration object.
        Config object should contain the following entries:
            - loss_function_classes (list): List of loss function class names to use.
            - loss_function_configs (list): List of configuration dictionaries for each 
                loss function.
            - data_split_classes (list): Name of the data splitting class to use. For 
                one-stage loss handler, this should be a list of length 1 with only one 
                split.
            - data_split_configs (list): List of configuration dictionaries for each 
                data splitting class.
            - LW_alg_classes (list): List of loss weighting algorithm class names to 
                use. For one-stage loss handler, this should be a list of length 1 with 
                only one algorithm.
            - LW_alg_configs (list): List of configuration dictionaries for each loss 
                weighting algorithm.
            - regularizer_classes (list, optional): List of regularizer class names to 
                use.
            - regularizer_configs (list, optional): List of configuration dictionaries 
                for each regularizer class. 
        """
        super(One_Stage_Loss_Handler, self).__init__()
        # Check that there is only one data split and one loss weighting algorithm
        if len(config["data_split_classes"]) != 1:
            raise ValueError(
                "One_Stage_Loss_Handler requires exactly one data split class."
                )
        if len(config["LW_alg_classes"]) != 1:
            raise ValueError(
                "One_Stage_Loss_Handler requires exactly one loss weighting algorithm class."
                )
        # Instantiate loss functions, data splits, and loss weighting algorithms
        self.loss_functions = nn.ModuleList(
            [LOSS_REGISTRY[loss_fn_class](**config) for loss_fn_class, config in zip(
                config["loss_function_classes"], config["loss_function_configs"]
            )]
        )
        self.data_splits = DATA_SPLIT_REGISTRY[config["data_split_classes"][0]](config["data_split_configs"][0])
        # Extract number of splits from the data split class
        self.num_splits = self.data_splits.num_splits.item()
        # Add number of loss terms to the config for the loss weighting algorithm
        config["LW_alg_configs"][0]["num_loss_terms"] = len(self.loss_functions) * self.num_splits
        # Instantiate the loss weighting algorithm
        self.loss_weighting_algorithm = LW_ALG_REGISTRY[config["LW_alg_classes"][0]](**config["LW_alg_configs"][0])
        # Instantiate regularizers if provided
        if "regularizer_classes" in config and "regularizer_configs" in config:
            self.regularizers = nn.ModuleList(
                [LOSS_REGISTRY[reg_class](**reg_cfg) for reg_class, reg_cfg in zip(
                    config["regularizer_classes"], config["regularizer_configs"]
                )]
            )

    def compute_loss_terms(self):
        """
        Computes the loss terms for the given data.
        Args:
            none
        Returns:
            list: List of loss tensors for each loss function and data split.
        """
        # Split data into individual components
        context_splits = self.data_splits(self.context)
        # Initialize list to store losses
        losses = []
        # Compute loss for each data split and each loss function
        for context in context_splits:
            for loss_fn in self.loss_functions:
                loss = loss_fn(context)
                losses.append(loss)
        # Store loss terms as regular attribute (not buffer) since it's temporary computation
        self.loss_terms = torch.stack(losses)
        # return losses  # TODO: Decide whether to return losses or not
    
    def update_loss_weights(self):
        """
        Updates the loss weights using the loss weighting algorithm.
        Args:
            model: Model from which to extract information (e.g., model.parameters).
            optimizer: Model optimizer from which to extract information.
        """
        if not hasattr(self, "loss_terms"):
            raise ValueError(
                "Loss terms have not been computed. Call compute_loss_terms() first."
            )
        self.loss_weighting_algorithm.update(self.loss_terms, self.context)

    def compute_loss(self):
        """
        Computes the final loss by applying the loss weighting algorithm to the 
        computed loss terms.
        Args:
            None
        Returns:
            torch.Tensor: The final weighted loss.
        """
        if not hasattr(self, "loss_terms"):
            raise ValueError(
                "Loss terms have not been computed. Call compute_loss_terms() first."
            )
        weighted_loss = self.loss_weighting_algorithm(self.loss_terms)
        # Apply regularizers if provided
        if hasattr(self, "regularizers"):
            for reg in self.regularizers:
                weighted_loss += reg(self.context)
        return weighted_loss


@register_loss_handler("Hierarchical_Loss_Handler")
class Hierarchical_Loss_Handler(Base_Loss_Handler):
    """
    Loss handler that computes loss hierarchically via two data splits. Useful for cases 
    in which there are two distinct ways to split data by task that should be handled 
    differently, e.g., split by source to correct data imbalances then split by output 
    and balance learning rate of each output.
    Assumes that each provided loss function corresponds to a separate task with respect 
    to each loss weighting algorithm. The first provided data split class further 
    differentiates tasks with respect to the first loss weighting algorithm, and the 
    same applies to the second data split class and the second loss weighting algorithm.
    """
    def __init__(self, config: dict[any, any]):
        """
        Initializes the Hierarchical_Loss_Handler with a configuration object.
        Config object should contain the following entries:
            - loss_function_classes (list): List of loss function class names to use.
            - loss_function_configs (list): List of configuration dictionaries for each 
                loss function.
            - data_split_classes (list): List of data splitting class names to use. 
                Should contain two classes for hierarchical splitting.
            - data_split_configs (list): List of configuration dictionaries for each 
                data splitting class.
            - LW_alg_classes (list): List of loss weighting algorithm class names to 
                use. Should contain two classes for hierarchical splitting, 
                corresponding to the two splits.
            - LW_alg_configs (list): List of configuration dictionaries for each loss 
                weighting algorithm.
            - regularizer_classes (list, optional): List of regularizer class names to 
                use.
            - regularizer_configs (list, optional): List of configuration dictionaries 
                for each regularizer class.
        """
        super(Hierarchical_Loss_Handler, self).__init__()
        # Check that there are two data splits and one loss weighting algorithm
        if len(config["data_split_classes"]) != 2:
            raise ValueError(
                "Hierarchical_Loss_Handler requires exactly two data split classes."
                )
        if len(config["LW_alg_classes"]) != 2:
            raise ValueError(
                "Hierarchical_Loss_Handler requires exactly two loss weighting algorithm classes."
                )
        # Instantiate loss functions, data splits, and loss weighting algorithms
        self.loss_functions = nn.ModuleList(
            [LOSS_REGISTRY[loss_fn_class](**loss_config) for loss_fn_class, loss_config in zip(
                config["loss_function_classes"], config["loss_function_configs"]
            )]
        )
        self.data_splits = nn.ModuleList(
            [DATA_SPLIT_REGISTRY[data_split_class](split_config) for data_split_class, split_config in zip(
                config["data_split_classes"], config["data_split_configs"]
            )]
        )
        # Extract number of splits from each data split class
        self.num_splits = [data_split.num_splits.item() for data_split in self.data_splits]
        # Set number of loss terms for each loss weighting algorithm
        for i in range(len(config["LW_alg_configs"])):
            config["LW_alg_configs"][i]["num_loss_terms"] = len(self.loss_functions) * self.num_splits[i]
        # Instantiate the loss weighting algorithms (nn.ModuleList so buffers are tracked by state_dict)
        self.loss_weighting_algorithm = nn.ModuleList(
            [LW_ALG_REGISTRY[config["LW_alg_classes"][i]](**config["LW_alg_configs"][i]) for i in range(len(config["LW_alg_classes"]))]
        )
        # Instantiate regularizers if provided
        if "regularizer_classes" in config and "regularizer_configs" in config:
            self.regularizers = nn.ModuleList(
                [LOSS_REGISTRY[reg_class](**reg_cfg) for reg_class, reg_cfg in zip(
                    config["regularizer_classes"], config["regularizer_configs"]
                )]
            )

    def compute_loss_terms(self):
            """
            Computes the loss terms for the given data.
            Args:
                none
            Returns:
                torch.Tensor: Hierarchical list containing a list of loss tensors for each loss 
                function and data split. Individual loss tensor lists are split by the 
                second data split, while the lists themselves are split by the first 
                data split.
            """
            # Split data into individual components hierarchically
            context_splits = []
            outer_splits = self.data_splits[1](self.context)  # First data split
            for context_split in outer_splits:
                context_splits.append(self.data_splits[0](context_split))  # Second data split
            # Initialize list to store losses and track non-empty splits
            losses = []
            non_empty_mask_list = []
            # Compute loss hierarchically for each data split and each loss function
            for context_list in context_splits:
                losses_list = []
                mask_list = []
                for context in context_list:
                    # Check if this split is non-empty
                    batch_size = context.batch['targets'].shape[0]
                    is_non_empty = batch_size > 0
                    # Replicate mask for each loss function
                    for loss_fn in self.loss_functions:
                        loss = loss_fn(context)
                        losses_list.append(loss)
                        mask_list.append(is_non_empty)
                losses.append(torch.stack(losses_list))
                non_empty_mask_list.append(torch.tensor(mask_list))
            losses = torch.stack(losses)  # Stack losses by outer split
            # Store loss terms and mask for empty splits as regular attributes (not buffers)
            # since they're temporary computation results and shouldn't be saved in checkpoints
            self.loss_terms = losses
            # Convert mask to tensor with same device and shape as loss_terms
            device = losses.device
            self.non_empty_mask = torch.stack(non_empty_mask_list).to(device)
            return losses  # TODO: Decide whether to return losses or not
        
    def update_loss_weights(self):
        """
        # TODO: Evaluate whether this is a good strategy for hierarchical loss 
        # weighting in general. Does this do what I want? Do the same for the 
        # compute_loss method.
        Updates the loss weights using the loss weighting algorithms.
        Args:
            none
        """
        if not hasattr(self, "loss_terms"):
            raise ValueError(
                "Loss terms have not been computed. Call compute_loss_terms() first."
            )
        # Mask out empty splits before updating weights to avoid biasing weight updates
        masked_loss_terms = self.loss_terms * self.non_empty_mask.float()
        # Update inner loss weighting algorithm by summing along the first dimension
        inner_loss_terms = torch.sum(masked_loss_terms, dim=0)
        self.loss_weighting_algorithm[0].update(inner_loss_terms, self.context)
        # Update outer loss weighting algorithm by summing along the second dimension
        outer_loss_terms = torch.sum(masked_loss_terms, dim=1)
        self.loss_weighting_algorithm[1].update(outer_loss_terms, self.context)

    def compute_loss(self):
        """
        Computes the final loss by applying the loss weighting algorithm to the 
        computed loss terms.
        Args:
            none
        Returns:
            torch.Tensor: The final weighted loss.
        """
        if not hasattr(self, "loss_terms"):
            raise ValueError(
                "Loss terms have not been computed. Call compute_loss_terms() first."
            )
        # Mask out empty splits (set to 0) to prevent them from contributing to loss
        # This ensures consistent loss values across batches with different source distributions
        masked_loss_terms = self.loss_terms * self.non_empty_mask.float()
        # Apply inner loss weighting algorithm to the loss terms by multiplying 
        # element-wise through each row of loss terms
        # Ensure weights are on the same device as loss_terms
        weights = self.loss_weighting_algorithm[0].weights.to(self.loss_terms.device)
        inner_weighted_loss = weights * masked_loss_terms
        inner_weighted_loss = torch.sum(inner_weighted_loss, dim=0)
        # Apply outer loss weighting algorithm
        weighted_loss = self.loss_weighting_algorithm[1](inner_weighted_loss)
        # Apply regularizers if provided
        if hasattr(self, "regularizers"):
            for reg in self.regularizers:
                weighted_loss += reg(self.context)
        return weighted_loss
    

# TODO: Should we include a "build loss handler" function that does a simplified construction process similar to the constructor for ProNDF?
# TODO: ^^ Likely sufficient to include some examples instead.
# def Build_Loss_Handler(
#     # Data and model parameters
#     dsource: int,
#     dout: int,
#     probabilistic_output: bool,
#     # Loss function parameters
#     loss_function_classes: list[str],
#     regularizer_classes: list[str]
#     LW_alg_classes: list[str],
#     heirarchical_weighting: bool = True,
#     
# ):
