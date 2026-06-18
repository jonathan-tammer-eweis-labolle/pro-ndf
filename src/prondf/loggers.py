"""
loggers.py
This module contains classes for logging training metrics to TensorBoard and other logging backends.
The logging system is modular - users can compose different loggers as needed without hardcoding
if statements throughout the model code.

Example usage:
    from loggers import (
        LossComponentsLogger, LossWeightsLogger, HyperparameterLogger, 
        LearningRateLogger, TruePredPlotLogger, LatentSpacePlotLogger
    )
    from Models import Build_ProNDF
    from data import load_splits
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_splits("path/to/data", "dataset_name")
    
    # Create loggers for training metrics
    train_loggers = [
        LossComponentsLogger(log_prefix="train", log_individual_terms=True),
        LossWeightsLogger(log_prefix="train", log_individual_weights=True),
        LearningRateLogger(log_prefix="train"),
    ]
    
    val_loggers = [
        LossComponentsLogger(log_prefix="val", log_individual_terms=False, log_sum=True, log_mean=True),
    ]
    
    # Create plot loggers (log every 5 epochs on validation to avoid slowing training)
    plot_loggers = [
        TruePredPlotLogger(
            dataset=val_dataset,
            log_prefix="val",
            log_frequency=5,  # Log every 5 epochs
            log_on_stage="val",
        ),
        LatentSpacePlotLogger(
            block_idx=0,  # Source latent space (B1)
            log_prefix="val",
            log_frequency=5,
            log_on_stage="val",
        ),
    ]
    
    # Combine all loggers
    all_loggers = train_loggers + val_loggers + plot_loggers
    
    # Create model with loggers
    model = Build_ProNDF(
        dsource=3,
        dcat=[],
        dnum=2,
        dout=1,
        qual_in=False,
        quant_in=True,
        loggers=all_loggers
    )
    
    # IMPORTANT: The custom loggers above use model.log() which requires a PyTorch Lightning
    # logger to be passed to the Trainer. You must create a TensorBoardLogger (or other PL logger)
    # and pass it to the Trainer.
    
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # Create TensorBoardLogger with custom save location (optional)
    tb_logger = TensorBoardLogger(
        save_dir="./logs",           # Directory to save logs (default: "lightning_logs")
        name="my_experiment",        # Experiment name (default: "lightning_logs")
        version=0                     # Version number (or None for auto-increment)
    )
    # Logs will be saved to: ./logs/my_experiment/version_0/
    
    # Create trainer with the TensorBoardLogger
    trainer = Trainer(
        logger=tb_logger,  # REQUIRED: Custom loggers use model.log() which needs a PL logger
        max_epochs=100,
        # ... other trainer args
    )
    
    # Train the model - custom loggers will automatically log via model.log()
    trainer.fit(model, train_loader, val_loader)
    
    # View logs in TensorBoard: tensorboard --logdir=./logs/my_experiment
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import torch
import pytorch_lightning as pl
import matplotlib
import sys

# Only set Agg backend if we're not in an interactive environment (like Jupyter)
# Agg is non-interactive but efficient for server environments and logging
# Check if we're in an interactive environment
is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules or 'IPython' in sys.modules

if not is_interactive:
    # Only set Agg backend if not in interactive environment
    # This allows Jupyter notebooks to use their default interactive backend
    try:
        matplotlib.use('Agg', force=False)
    except:
        pass  # Backend may already be set, ignore

import matplotlib.pyplot as plt
import numpy as np
from .data import MultiFidelityDataset


class BaseLogger(ABC):
    """
    Base class for all loggers. Defines the interface that all loggers must implement.
    """
    def __init__(self, log_prefix: str = "", on_step: bool = False, on_epoch: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names (e.g., "train", "val")
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
        """
        self.log_prefix = log_prefix
        self.on_step = on_step
        self.on_epoch = on_epoch
    
    def _get_log_name(self, metric_name: str) -> str:
        """Helper to construct full metric name with prefix."""
        if self.log_prefix:
            return f"{self.log_prefix}/{metric_name}"
        return metric_name
    
    @abstractmethod
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """
        Log metrics for the given model and stage.
        
        Args:
            model: The PyTorch Lightning model instance
            stage: The training stage ("train" or "val")
        """
        pass


class LossComponentsLogger(BaseLogger):
    """
    Logger for individual loss components (loss terms before weighting).
    Logs each element of the loss_terms tensor from the loss handler.
    """
    def __init__(
        self,
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        log_individual_terms: bool = True,
        log_sum: bool = True,
        log_mean: bool = True,
        name_format: str = "loss_term_{idx}"
    ):
        """
        Initialize the loss components logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
            log_individual_terms: Whether to log each individual loss term
            log_sum: Whether to log the sum of all loss terms
            log_mean: Whether to log the mean of all loss terms
            name_format: Format string for naming individual loss terms (use {idx} for index)
        """
        super().__init__(log_prefix, on_step, on_epoch)
        self.log_individual_terms = log_individual_terms
        self.log_sum = log_sum
        self.log_mean = log_mean
        self.name_format = name_format
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log loss components from the loss handler."""
        if not hasattr(model.loss_handler, 'loss_terms'):
            return  # Loss terms not computed yet
        
        loss_terms = model.loss_handler.loss_terms
        
        # Handle both 1D (One_Stage) and 2D (Hierarchical) loss terms
        if loss_terms.dim() == 2:
            # Hierarchical loss handler - flatten for summary stats
            loss_terms_flat = loss_terms.flatten()
        else:
            loss_terms_flat = loss_terms
        
        if self.log_sum:
            total = torch.sum(loss_terms_flat)
            model.log(
                self._get_log_name("loss_components/sum"),
                total,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                prog_bar=False
            )
        
        if self.log_mean:
            mean = torch.mean(loss_terms_flat)
            model.log(
                self._get_log_name("loss_components/mean"),
                mean,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                prog_bar=False
            )
        
        if self.log_individual_terms:
            # For hierarchical, log with 2D indexing
            if loss_terms.dim() == 2:
                for i in range(loss_terms.shape[0]):
                    for j in range(loss_terms.shape[1]):
                        term_name = self.name_format.format(idx=f"{i}_{j}")
                        model.log(
                            self._get_log_name(f"loss_components/{term_name}"),
                            loss_terms[i, j],
                            on_step=self.on_step,
                            on_epoch=self.on_epoch,
                            prog_bar=False
                        )
            else:
                for idx, term in enumerate(loss_terms):
                    term_name = self.name_format.format(idx=idx)
                    model.log(
                        self._get_log_name(f"loss_components/{term_name}"),
                        term,
                        on_step=self.on_step,
                        on_epoch=self.on_epoch,
                        prog_bar=False
                    )


class LossWeightsLogger(BaseLogger):
    """
    Logger for loss weights from loss weighting algorithms.
    Logs the weights used by the loss weighting algorithm(s).
    """
    def __init__(
        self,
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        log_individual_weights: bool = True,
        log_sum: bool = True,
        log_mean: bool = True,
        name_format: str = "weight_{idx}"
    ):
        """
        Initialize the loss weights logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
            log_individual_weights: Whether to log each individual weight
            log_sum: Whether to log the sum of all weights
            log_mean: Whether to log the mean of all weights
            name_format: Format string for naming individual weights (use {idx} for index)
        """
        super().__init__(log_prefix, on_step, on_epoch)
        self.log_individual_weights = log_individual_weights
        self.log_sum = log_sum
        self.log_mean = log_mean
        self.name_format = name_format
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log loss weights from the loss weighting algorithm(s)."""
        loss_handler = model.loss_handler
        
        # Handle different loss handler types
        if hasattr(loss_handler, 'loss_weighting_algorithm'):
            import torch.nn as nn
            if isinstance(loss_handler.loss_weighting_algorithm, nn.ModuleList):
                # Hierarchical_Loss_Handler with multiple algorithms
                for idx, alg in enumerate(loss_handler.loss_weighting_algorithm):
                    self._log_weights_from_algorithm(
                        model,
                        alg,
                        f"loss_weights/stage_{idx}"
                    )
            else:
                # One_Stage_Loss_Handler or single algorithm
                self._log_weights_from_algorithm(
                    model,
                    loss_handler.loss_weighting_algorithm,
                    "loss_weights"
                )
    
    def _log_weights_from_algorithm(
        self,
        model: pl.LightningModule,
        algorithm,
        prefix: str
    ):
        """Helper to log weights from a single loss weighting algorithm."""
        if not hasattr(algorithm, 'weights'):
            return  # Algorithm doesn't have weights to log
        
        weights = algorithm.weights
        
        if self.log_sum:
            total = torch.sum(weights)
            model.log(
                self._get_log_name(f"{prefix}/sum"),
                total,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                prog_bar=False
            )
        
        if self.log_mean:
            mean = torch.mean(weights)
            model.log(
                self._get_log_name(f"{prefix}/mean"),
                mean,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                prog_bar=False
            )
        
        if self.log_individual_weights:
            for idx, weight in enumerate(weights):
                weight_name = self.name_format.format(idx=idx)
                model.log(
                    self._get_log_name(f"{prefix}/{weight_name}"),
                    weight,
                    on_step=self.on_step,
                    on_epoch=self.on_epoch,
                    prog_bar=False
                )


class RegularizerLogger(BaseLogger):
    """
    Logger for regularizer values.
    Logs the value of each regularizer separately.
    """
    def __init__(
        self,
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        name_format: str = "regularizer_{idx}"
    ):
        """
        Initialize the regularizer logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
            name_format: Format string for naming regularizers (use {idx} for index)
        """
        super().__init__(log_prefix, on_step, on_epoch)
        self.name_format = name_format
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log regularizer values."""
        loss_handler = model.loss_handler
        
        if not hasattr(loss_handler, 'regularizers'):
            return  # No regularizers
        
        # Compute regularizer values
        for idx, regularizer in enumerate(loss_handler.regularizers):
            reg_value = regularizer(loss_handler.context)
            reg_name = self.name_format.format(idx=idx)
            model.log(
                self._get_log_name(f"regularizers/{reg_name}"),
                reg_value,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                prog_bar=False
            )


class HyperparameterLogger(BaseLogger):
    """
    Logger for model hyperparameters.
    Logs specified hyperparameters as scalars (useful for tracking learning rate schedules, etc.).
    """
    def __init__(
        self,
        hyperparams_to_log: List[str],
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True
    ):
        """
        Initialize the hyperparameter logger.
        
        Args:
            hyperparams_to_log: List of hyperparameter names to log (e.g., ["lr", "weight_decay"])
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
        """
        super().__init__(log_prefix, on_step, on_epoch)
        self.hyperparams_to_log = hyperparams_to_log
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log specified hyperparameters."""
        if not hasattr(model, 'hparams'):
            return  # No hyperparameters saved
        
        hparams = model.hparams
        
        for param_name in self.hyperparams_to_log:
            if hasattr(hparams, param_name):
                param_value = getattr(hparams, param_name)
                # Only log scalar values
                if isinstance(param_value, (int, float)):
                    model.log(
                        self._get_log_name(f"hyperparams/{param_name}"),
                        float(param_value),
                        on_step=self.on_step,
                        on_epoch=self.on_epoch,
                        prog_bar=False
                    )
            elif isinstance(hparams, dict) and param_name in hparams:
                param_value = hparams[param_name]
                if isinstance(param_value, (int, float)):
                    model.log(
                        self._get_log_name(f"hyperparams/{param_name}"),
                        float(param_value),
                        on_step=self.on_step,
                        on_epoch=self.on_epoch,
                        prog_bar=False
                    )


class LearningRateLogger(BaseLogger):
    """
    Logger for learning rate(s) from the optimizer.
    Useful for tracking learning rate schedules.
    """
    def __init__(
        self,
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True
    ):
        """
        Initialize the learning rate logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step
            on_epoch: Whether to log on each epoch
        """
        super().__init__(log_prefix, on_step, on_epoch)
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log learning rate(s) from the optimizer."""
        # Get optimizer - PyTorch Lightning's optimizers() method returns the optimizer
        optimizer = model.optimizers()
        
        # Handle both single optimizer and list of optimizers
        if not isinstance(optimizer, (list, tuple)):
            optimizers = [optimizer]
        else:
            optimizers = optimizer
        
        for opt_idx, opt in enumerate(optimizers):
            # Get learning rate(s) from optimizer
            if hasattr(opt, 'param_groups'):
                for group_idx, param_group in enumerate(opt.param_groups):
                    if 'lr' in param_group:
                        lr = param_group['lr']
                        if len(optimizers) > 1 or len(opt.param_groups) > 1:
                            name = f"learning_rate/optimizer_{opt_idx}_group_{group_idx}"
                        else:
                            name = "learning_rate"
                        model.log(
                            self._get_log_name(name),
                            lr,
                            on_step=self.on_step,
                            on_epoch=self.on_epoch,
                            prog_bar=False
                        )


class PlotLogger(BaseLogger):
    """
    Base class for plot loggers. Provides efficient figure-to-tensor conversion
    and periodic logging to avoid slowing down training.
    """
    def __init__(
        self,
        log_prefix: str = "",
        on_step: bool = False,
        on_epoch: bool = True,
        log_frequency: int = 1,
        log_on_stage: Optional[str] = None,
        dataset: Optional[MultiFidelityDataset] = None,
        device: str = 'cpu',
        batch_size: int = 32,
        plot_name: str = "plot",
    ):
        """
        Initialize the plot logger.
        
        Args:
            log_prefix: Prefix to add to all logged metric names
            on_step: Whether to log on each training step (ignored for plots, use log_frequency instead)
            on_epoch: Whether to log on each epoch (ignored for plots, use log_frequency instead)
            log_frequency: Log plots every N epochs (if on_epoch=True) or every N steps (if on_step=True)
            log_on_stage: Stage to log on ("train", "val", or None for both). Defaults to "val" for efficiency.
            dataset: Dataset to use for generating plots (required for most plot types)
            device: Device to run model on
            batch_size: Batch size for predictions
            plot_name: Name for the plot in TensorBoard
        """
        super().__init__(log_prefix, on_step, on_epoch)
        self.log_frequency = log_frequency
        self.log_on_stage = log_on_stage if log_on_stage is not None else "val"
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.plot_name = plot_name
        self._step_count = 0
        self._last_logged_epoch = -1  # Track last epoch we logged on
        self._last_logged_step = -1   # Track last step we logged on
    
    def _should_log(self, model: pl.LightningModule, stage: str, is_step: bool) -> bool:
        """Check if we should log at this step/epoch."""
        # First check if we should log on this stage
        if self.log_on_stage and stage != self.log_on_stage:
            return False
        
        # Now check frequency - use model's epoch/step tracking instead of our own counter
        if is_step:
            # For step logging, use global_step
            current_step = model.global_step
            if current_step != self._last_logged_step:
                self._last_logged_step = current_step
                return (current_step % self.log_frequency == 0)
            return False
        else:
            # For epoch logging, use current_epoch and only log once per epoch
            current_epoch = model.current_epoch
            if current_epoch != self._last_logged_epoch:
                self._last_logged_epoch = current_epoch
                return (current_epoch % self.log_frequency == 0)
            return False
    
    def _fig_to_tensor(self, fig) -> torch.Tensor:
        """
        Efficiently convert matplotlib figure to tensor for TensorBoard.
        
        Args:
            fig: matplotlib figure object
            
        Returns:
            torch.Tensor: Image tensor of shape (C, H, W) in RGB format
        """
        # Draw the figure to render it
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure (.copy() makes it writable for torch.from_numpy)
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()

        # Convert to tensor and change from HWC to CHW format
        img_tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
        
        # Close figure to free memory
        plt.close(fig)
        
        return img_tensor
    
    @abstractmethod
    def _create_plot(self, model: pl.LightningModule) -> plt.Figure:
        """
        Create the plot figure. Must be implemented by subclasses.
        
        Args:
            model: The PyTorch Lightning model instance
            
        Returns:
            matplotlib figure object
        """
        pass
    
    def log(self, model: pl.LightningModule, stage: str = "train"):
        """Log plot to TensorBoard."""
        # Check if we should log
        is_step = self.on_step and stage == "train"
        is_epoch = self.on_epoch and not is_step
        
        if not self._should_log(model, stage, is_step):
            return
        
        # Only check for dataset if this logger requires one
        # Some loggers (like LatentSpacePlotLogger) don't need a dataset
        requires_dataset = getattr(self, '_requires_dataset', True)  # Default to True for backward compatibility
        if requires_dataset and self.dataset is None:
            return  # Can't create plot without dataset
        
        try:
            # Create plot
            fig = self._create_plot(model)
            
            # Convert to tensor
            img_tensor = self._fig_to_tensor(fig)
            
            # Log to TensorBoard using logger.experiment.add_image()
            # This is the standard way to log images to TensorBoard in PyTorch Lightning
            if hasattr(model, 'logger') and model.logger is not None:
                # For TensorBoard, we should use global_step (training step) not epoch
                # This ensures images appear correctly in TensorBoard
                global_step = model.global_step
                
                # Get the tag name
                tag = self._get_log_name(self.plot_name)
                
                # Log image directly to TensorBoard
                # img_tensor should be shape (C, H, W) which is what we have from _fig_to_tensor
                model.logger.experiment.add_image(
                    tag=tag,
                    img_tensor=img_tensor,
                    global_step=global_step
                )
            else:
                # Fallback: try model.log() if logger.experiment is not available
                # This shouldn't normally be needed, but provides a fallback
                model.log(
                    self._get_log_name(self.plot_name),
                    img_tensor,
                    on_step=is_step,
                    on_epoch=is_epoch,
                    prog_bar=False
                )
        except Exception as e:
            # Log the error instead of silently failing, but don't break training
            import warnings
            import traceback
            warnings.warn(
                f"Failed to log plot '{self.plot_name}': {str(e)}\n"
                f"Traceback: {traceback.format_exc()}",
                UserWarning
            )


class TruePredPlotLogger(PlotLogger):
    """
    Logger for true vs predicted scatter plots.
    Logs plots showing model predictions vs true values for each data source.
    """
    def __init__(
        self,
        dataset: MultiFidelityDataset,
        log_prefix: str = "",
        on_epoch: bool = True,
        log_frequency: int = 5,
        log_on_stage: str = "val",
        device: str = 'cpu',
        batch_size: int = 32,
        colors: tuple = ("red", "blue"),
        lw: int = 2,
        s: int = 30,
        figsize: int = 7,
        noise_variance: Optional[float] = None,
    ):
        """
        Initialize the true-pred plot logger.
        
        Args:
            dataset: Dataset to use for generating plots
            log_prefix: Prefix to add to all logged metric names
            on_epoch: Whether to log on epochs (recommended True for plots)
            log_frequency: Log plots every N epochs
            log_on_stage: Stage to log on ("train", "val", or None for both). Defaults to "val".
            device: Device to run model on
            batch_size: Batch size for predictions
            colors: Tuple of (line_color, scatter_color) for the diagonal line and scatter points
            lw: Line width for diagonal line
            s: Scatter point size
            figsize: Base figure size
            noise_variance: Optional noise variance value(s) for noise floor visualization
        """
        super().__init__(
            log_prefix=log_prefix,
            on_step=False,  # Plots should log on epoch, not step
            on_epoch=on_epoch,
            log_frequency=log_frequency,
            log_on_stage=log_on_stage,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            plot_name="true_pred_plot",
        )
        self._requires_dataset = True  # TruePredPlotLogger requires a dataset
        self.colors = colors
        self.lw = lw
        self.s = s
        self.figsize = figsize
        self.noise_variance = noise_variance
    
    def _create_plot(self, model: pl.LightningModule) -> plt.Figure:
        """Create true vs predicted plot."""
        from .plotting import plot_true_pred
        # Note: noise_variance parameter was removed from plot_true_pred
        # If noise floor visualization is needed, it should be handled differently
        return plot_true_pred(
            model=model,
            test_dataset=self.dataset,
            device=self.device,
            batch_size=self.batch_size,
            colors=self.colors,
            lw=self.lw,
            s=self.s,
            figsize=self.figsize,
        )


class LatentSpacePlotLogger(PlotLogger):
    """
    Logger for 2D latent space plots.
    Logs plots showing the latent space representation for source block (B1) or categorical block (B2).
    """
    def __init__(
        self,
        block_idx: int,
        dcat: Optional[List[int]] = None,
        log_prefix: str = "",
        on_epoch: bool = True,
        log_frequency: int = 5,
        log_on_stage: str = "val",
        device: str = 'cpu',
        figsize: int = 6,
        num_iterations: int = 100,
    ):
        """
        Initialize the latent space plot logger.
        
        Args:
            block_idx: 0 for source latent space (B1), 1 for categorical latent space (B2)
            dcat: List of number of categories for each categorical input (required for B2)
            log_prefix: Prefix to add to all logged metric names
            on_epoch: Whether to log on epochs (recommended True for plots)
            log_frequency: Log plots every N epochs
            log_on_stage: Stage to log on ("train", "val", or None for both). Defaults to "val".
            device: Device to run model on
            figsize: Base figure size
            num_iterations: Number of samples to draw for probabilistic latent spaces
        """
        super().__init__(
            log_prefix=log_prefix,
            on_step=False,  # Plots should log on epoch, not step
            on_epoch=on_epoch,
            log_frequency=log_frequency,
            log_on_stage=log_on_stage,
            dataset=None,  # Latent space plots don't need a dataset
            device=device,
            batch_size=1,  # Not used for latent space plots
            plot_name=f"latent_space_B{block_idx + 1}_plot",
        )
        self._requires_dataset = False  # LatentSpacePlotLogger doesn't require a dataset
        self.block_idx = block_idx
        self.dcat = dcat
        self.figsize = figsize
        self.num_iterations = num_iterations
    
    def _create_plot(self, model: pl.LightningModule) -> plt.Figure:
        """Create latent space plot."""
        from .plotting import plot_2D_latent_space
        return plot_2D_latent_space(
            model=model,
            block_idx=self.block_idx,
            dcat=self.dcat,
            device=self.device,
            figsize=self.figsize,
            num_iterations=self.num_iterations,
        )


# Registry for loggers (similar to other registries in the codebase)
LOGGER_REGISTRY = {}

def register_logger(name: str):
    """
    Decorator to register a logger class with the given name.
    
    Args:
        name: The name of the logger type to register.
    
    Returns:
        function: The decorator function that registers the logger.
    """
    def decorator(cls):
        LOGGER_REGISTRY[name] = cls
        return cls
    return decorator


