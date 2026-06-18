"""
Model definitions and constructors for ProNDF.
"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import warnings
from typing import Optional, List
from .blocks import BLOCK_REGISTRY
from .losses import LOSS_HANDLER_REGISTRY
from .optimizers import OPTIMIZER_REGISTRY
from .loggers import LearningRateLogger


class ProNDF(pl.LightningModule):
    """
    Probabilistic neural network for multi-fidelity data fusion.
    
    This LightningModule composes three blocks (source, categorical, and output)
    and delegates loss computation to a configurable loss handler.
    """
    def __init__(
        self,
        # Data parameters
        dsource: int,  # the number of data sources
        dcat: list[int],  # the number of categories for each categorical input
        dnum: int,  # the dimension of the numerical input
        dout: int,  # the dimenson of the output
        qual_in: bool,  # whether qualitative (categorical) inputs are present
        quant_in: bool,  # whether quantitative (numerical) inputs are present
        # Block / architecture parameters
        B1_type: str,  # Source block
        B1_config: dict[str, any],  # Block 1 parameters
        B2_type: str,  # Categorical block
        B2_config: dict[str, any],  # Block 2 parameters
        B3_type: str,  # Input-output relationship
        B3_config: dict[str, any],  # Block 3 parameters
        # Training parameters such as loss function, optimizer, and regularizers
        loss_handler_type: str,  # Loss handler to use
        loss_handler_config: dict[str, any],  # Loss handler config including loss functions and regularizers
        optimizer_type: str,  # Optimizer choice
        optimizer_config: dict[str, any],  # Optimizer configuration
        # Logging and other misc
        loggers: Optional[List] = None,  # List of logger instances for tracking training metrics
    ):
        """
        Initializes the ProNDF LightningModule and builds blocks, loss handler, and loggers.
        
        Args:
            dsource: Number of data sources.
            dcat: List of categorical levels per categorical input.
            dnum: Dimension of numerical inputs.
            dout: Dimension of outputs/targets.
            qual_in: Whether categorical inputs are present.
            quant_in: Whether numerical inputs are present.
            B1_type: Registry name for the source block.
            B1_config: Configuration for the source block.
            B2_type: Registry name for the categorical block.
            B2_config: Configuration for the categorical block.
            B3_type: Registry name for the input-output block.
            B3_config: Configuration for the input-output block.
            loss_handler_type: Registry name for the loss handler.
            loss_handler_config: Configuration for the loss handler.
            optimizer_type: Registry name for the optimizer.
            optimizer_config: Configuration for the optimizer.
            loggers: Optional list of custom logger instances.
        """
        super(ProNDF, self).__init__()
        # Sanity checks for inputs and configuration
        if not isinstance(dsource, int) or dsource <= 0:
            raise ValueError("dsource must be a positive integer.")
        if not isinstance(dout, int) or dout <= 0:
            raise ValueError("dout must be a positive integer.")
        if not isinstance(qual_in, bool) or not isinstance(quant_in, bool):
            raise TypeError("qual_in and quant_in must be booleans.")
        if quant_in:
            if not isinstance(dnum, int) or dnum <= 0:
                raise ValueError("dnum must be a positive integer when quant_in=True.")
        else:
            if dnum not in (0, None):
                raise ValueError("dnum must be 0 or None when quant_in=False.")
        if qual_in:
            if not isinstance(dcat, list) or len(dcat) == 0:
                raise ValueError("dcat must be a non-empty list when qual_in=True.")
            if not all(isinstance(level, int) and level > 0 for level in dcat):
                raise ValueError("dcat must contain positive integers.")
        if not isinstance(B1_config, dict) or not isinstance(B2_config, dict) or not isinstance(B3_config, dict):
            raise TypeError("B1_config, B2_config, and B3_config must be dictionaries.")
        if not isinstance(loss_handler_config, dict) or not isinstance(optimizer_config, dict):
            raise TypeError("loss_handler_config and optimizer_config must be dictionaries.")
        if B1_type not in BLOCK_REGISTRY:
            raise KeyError(f"Unknown B1_type '{B1_type}'.")
        if B3_type not in BLOCK_REGISTRY:
            raise KeyError(f"Unknown B3_type '{B3_type}'.")
        if qual_in and B2_type not in BLOCK_REGISTRY:
            raise KeyError(f"Unknown B2_type '{B2_type}'.")
        if loss_handler_type not in LOSS_HANDLER_REGISTRY:
            raise KeyError(f"Unknown loss_handler_type '{loss_handler_type}'.")
        if optimizer_type not in OPTIMIZER_REGISTRY:
            raise KeyError(f"Unknown optimizer_type '{optimizer_type}'.")
        if loggers is not None:
            if not isinstance(loggers, list):
                raise TypeError("loggers must be a list or None.")
            for logger in loggers:
                if not hasattr(logger, "log"):
                    raise TypeError("Each logger must implement a log(model, stage=...) method.")
        # Save parameters (exclude loggers from hyperparameters as they're not serializable)
        self.save_hyperparameters(ignore=['loggers'])
        # Build blocks and loss handler
        # Build source block
        self.B1 = BLOCK_REGISTRY[B1_type](**B1_config)
        # Build categorical block if necessary
        if qual_in:
            self.B2 = BLOCK_REGISTRY[B2_type](**B2_config)
        # Build input-output block
        self.B3 = BLOCK_REGISTRY[B3_type](**B3_config)
        # Build loss handler
        self.loss_handler = LOSS_HANDLER_REGISTRY[loss_handler_type](loss_handler_config)
        # Store loggers (use _loggers to avoid conflict with PyTorch Lightning's property system)
        self._loggers = loggers if loggers is not None else []
    
    @property
    def loggers(self):
        """Property to access loggers (for backward compatibility)."""
        return self._loggers

    def forward(self, batch):
        """
        Performs a forward pass and returns model output tensor.
        
        If block 3 is probabilistic, this returns a sample from the output distribution.
        
        Args:
            batch: Dictionary with keys 'source', 'cat', 'num', and 'targets'.
        
        Returns:
            torch.Tensor: Model outputs for the batch.
        """
        source = batch['source']
        cat = batch['cat']
        num = batch['num']
        # Get source manifold
        z_B1 = self.B1(source)
        # Get categorical manifold if necessary
        if self.hparams.qual_in:
            z_B2 = self.B2(cat)
        # Concatenate as necessary and pass through block 3
        # Checks to avoid iterative torch.cat operations
        if self.hparams.qual_in and self.hparams.quant_in:  # Both qual and quant inputs
            u = torch.cat((z_B1, z_B2, num), dim = -1)  # u is combined input
        elif self.hparams.qual_in:  # Only qual inputs
            u = torch.cat((z_B1, z_B2), dim = -1)  # u is combined input
        else:  # Only quant inputs
            u = torch.cat((z_B1, num), dim = -1)  # u is combined input
        out = self.B3(u)
        return out
        
    def get_model_outputs(self, batch):
        """
        Builds output dictionary for loss handling and logging.
        
        Args:
            batch: Dictionary with keys 'source', 'cat', 'num', and 'targets'.
        
        Returns:
            dict: Nested dict of outputs for each block, including distributions
                when blocks are probabilistic.
        """
        source = batch['source']
        cat = batch['cat']
        num = batch['num']
        outputs = {}
        # Get source manifold
        B1_outputs = {}
        z_B1 = self.B1(source)  # u is combined input to be concatenated
        B1_outputs["out"] = z_B1
        if self.B1.probabilistic_output:
            z_B1_dist = self.B1.predict_distribution(source)
            B1_outputs["out_dist"] = z_B1_dist
        outputs["B1"] = B1_outputs
        # Get categorical manifold if necessary
        if self.hparams.qual_in:
            B2_outputs = {}
            z_B2 = self.B2(cat)
            B2_outputs["out"] = z_B2
            if self.B2.probabilistic_output:
                z_B2_dist = self.B2.predict_distribution(cat)
                B2_outputs["out_dist"] = z_B2_dist
            outputs["B2"] = B2_outputs
        # Concatenate as necessary and pass through block 3
        # Checks to avoid iterative torch.cat operations
        if self.hparams.qual_in and self.hparams.quant_in:  # Both qual and quant inputs
            u = torch.cat((z_B1, z_B2, num), dim = -1)
        elif self.hparams.qual_in:  # Only qual inputs
            u = torch.cat((z_B1, z_B2), dim = -1)
        else:  # Only quant inputs
            u = torch.cat((z_B1, num), dim = -1)
        B3_outputs = {}
        out = self.B3(u)
        B3_outputs["out"] = out
        if self.B3.probabilistic_output:
            z_B3_dist = self.B3.predict_distribution(u)
            B3_outputs["out_dist"] = z_B3_dist
        outputs["B3"] = B3_outputs
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Model forward pass, loss term calculations, and loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Update loss weights
        self.loss_handler.update_loss_weights()
        # Compute final loss
        loss = self.loss_handler.compute_loss()
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # Call custom loggers
        for logger in self._loggers:
            logger.log(self, stage="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Model forward pass and loss term calculations. No loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Compute final loss
        loss = self.loss_handler.compute_loss()
        # Log validation loss for early stopping and monitoring
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Call custom loggers
        for logger in self._loggers:
            logger.log(self, stage="val")
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Model forward pass and loss term calculations. No loss weight updates
        """
        # Get model outputs
        outputs = self.get_model_outputs(batch)
        # Build loss context
        self.loss_handler.build_loss_context(self, batch, outputs)
        # Compute loss terms for weighting
        self.loss_handler.compute_loss_terms()
        # Compute final loss including regularization
        loss = self.loss_handler.compute_loss()
        return loss
    
    def configure_optimizers(self):
        """
        Initializes and configures optimizers using provided parameters.
        Note: Model parameters are passed here since they're only available after model initialization.
        """
        optimizer = OPTIMIZER_REGISTRY[self.hparams.optimizer_type](
            self.parameters(), 
            **self.hparams.optimizer_config
        )
        return optimizer
    

def Build_ProNDF(
    dataset_meta: dict,  # Dataset metadata dictionary containing dsource, dcat, dnum, dtargets, qual_in, quant_in
    # Architecture and block parameters
    dz_B1: int = 2,
    dz_B2: int = 2,
    architecture: dict[str, list[int]] = {
        "B1": [8, 4],
        "B2": [8, 4],
        "B3": [16, 32, 16, 8],
        },
    hidden_act_fn = "Tanh",
    output_act_fn = "Identity",
    probabilistic_manifolds = False,
    probabilistic_output = True,
    # Optimizer and regularizer params
    lr: float = 0.001,
    weight_decay_strength: float = 0.001,
    regularizer_strength: float = 0.1,
    # Loss weighting
    # True - heirarchical loss weighting w/ two-moment weighting algorithm
    # False - one-stage loss weighting w/ no weighting algorithm
    loss_weighting: bool = True,
    # Logging
    loggers: Optional[List] = None,  # List of logger instances for tracking training metrics
):
    """
    Streamlined constructor for ProNDF including basic functionality. For more 
    flexibility and advanced usage, initialize the model directly using the ProNDF 
    class with appropriate config dictionaries.
    
    Args:
        dataset_meta: Dictionary containing dataset metadata with keys:
            - dsource: Number of data sources
            - dcat: List of number of categories for each categorical input (or None/empty list)
            - dnum: Dimension of the numerical input
            - dtargets: Dimension of the output/targets
            - qual_in: Whether qualitative (categorical) inputs are present
            - quant_in: Whether quantitative (numerical) inputs are present
        Other arguments: See function signature for architecture and training parameters.
    
    Returns:
        ProNDF: Configured ProNDF model instance.
    """
    if loggers is None:
        loggers = [LearningRateLogger()]
    # Extract data parameters from dataset meta
    dsource = dataset_meta['dsource']
    dcat = dataset_meta.get('dcat', None)
    dnum = dataset_meta.get('dnum', None)
    dout = dataset_meta['dtargets']
    qual_in = dataset_meta['qual_in']
    quant_in = dataset_meta['quant_in']
    # Build configs for each model component
    # Blocks 1 and 2 types
    if probabilistic_manifolds:
        B1_type = "Prob_Block"
        B2_type = "Prob_Block"
    else:
        B1_type = "Det_Block"
        B2_type = "Det_Block"
    # Block 3 type
    if probabilistic_output:
        B3_type = "Prob_Block"
    else:
        B3_type = "Det_Block"
    # Block 1 config (always needed)
    B1_config = {
        "d_in": dsource,
        "d_out": dz_B1,
        "hidden_layers": architecture["B1"],
        "hidden_act_fn": hidden_act_fn,
        "output_act_fn": "Identity",
    }
    # Block 2 config (only needed if qual_in is True)
    if qual_in:
        # Handle case where dcat might be None or empty
        if dcat is None or len(dcat) == 0:
            raise ValueError("dcat must be provided and non-empty when qual_in=True")
        B2_config = {
            "d_in": sum(dcat),
            "d_out": dz_B2,
            "hidden_layers": architecture["B2"],
            "hidden_act_fn": hidden_act_fn,
            "output_act_fn": "Identity",
        }
    else:
        # B2_config won't be used, but we need to provide something
        # The ProNDF class will check qual_in before using it
        B2_config = {
            "d_in": 0,  # Dummy value, won't be used
            "d_out": dz_B2,
            "hidden_layers": architecture["B2"],
            "hidden_act_fn": hidden_act_fn,
            "output_act_fn": "Identity",
        }
    # Block 3 config
    if qual_in and quant_in:  # Get combined input dimensionality
        d_u = dz_B1 + dz_B2 + dnum
    elif qual_in:
        d_u = dz_B1 + dz_B2
    else:
        d_u = dz_B1 + dnum
    B3_config = {
        "d_in": d_u,
        "d_out": dout,
        "hidden_layers": architecture["B3"],
        "hidden_act_fn": hidden_act_fn,
        "output_act_fn": output_act_fn,
    }
    # Loss functions and regularizers
    if probabilistic_output:
        loss_function_classes = ["Output_NLL_loss"]
        loss_function_configs = [{}]
        regularizer_classes = ["Output_IS_loss"]
        regularizer_configs = [{"alpha": 0.05, "strength": regularizer_strength}]
    else:
        loss_function_classes = ["Output_MSE_loss"]
        loss_function_configs = [{}]
        regularizer_classes = []
        regularizer_configs = []
    # Data splits and loss-weighting algorithms
    if loss_weighting:
        data_split_classes = ["Split_by_Source", "Split_by_Output_Dim"]
        data_split_configs = [{"num_sources": dsource}, {"num_outputs": dout}]
        LW_alg_classes = ["Fixed_Weights", "Two_Moment_Weighting"]
        LW_alg_configs = [{"num_loss_terms": dsource}, {"num_loss_terms": dout}]
        loss_handler_type = "Hierarchical_Loss_Handler"
    else:
        data_split_classes = ["No_Split"]
        data_split_configs = [{}]
        LW_alg_classes = ["No_Weighting"]
        LW_alg_configs = [{}]
        loss_handler_type = "One_Stage_Loss_Handler"
    # Build loss handler config
    loss_handler_config = {
            "loss_function_classes": loss_function_classes,
            "loss_function_configs": loss_function_configs,
            "data_split_classes": data_split_classes,
            "data_split_configs": data_split_configs,
            "LW_alg_classes": LW_alg_classes,
            "LW_alg_configs": LW_alg_configs,
            "regularizer_classes": regularizer_classes,
            "regularizer_configs": regularizer_configs,
        }
    # Optimizer type and config
    optimizer_type = "Adam"
    optimizer_config = {"lr": lr, "weight_decay": weight_decay_strength}
    # Build model
    model = ProNDF(
        dsource = dsource,
        dcat = dcat,
        dnum = dnum,
        dout = dout,
        qual_in = qual_in,
        quant_in = quant_in,
        B1_type = B1_type,
        B1_config = B1_config,
        B2_type = B2_type,
        B2_config = B2_config,
        B3_type = B3_type,
        B3_config = B3_config,
        loss_handler_type = loss_handler_type,
        loss_handler_config = loss_handler_config,
        optimizer_type = optimizer_type,
        optimizer_config = optimizer_config,
        loggers = loggers,
    )
    return model
