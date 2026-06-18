"""
Plotting utilities for ProNDF models and datasets.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import itertools
from torch.utils.data import DataLoader
from .data import to_categorical, MultiFidelityDataset, collate_fn

my_colors = [
    "red",
    "blue",
    "green",
    "magenta",
    "cyan",
    "yellow",
    "black",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "navy",
    "gold",
    "mediumslateblue",
    "maroon",
    "darkgoldenrod",
]  # List of colors to be used in plots

# For backward compatibility, alias the collate_fn from data module
_collate_fn = collate_fn


def _get_predictions(model, dataset, device, batch_size=32):
    """
    Helper function to get model predictions for a dataset.
    Returns predictions and optionally distribution parameters if probabilistic.
    """
    model.eval()
    model = model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)
    
    all_preds = []
    all_preds_dist = None
    is_probabilistic = model.B3.probabilistic_output
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.get_model_outputs(batch)
            
            if is_probabilistic:
                pred_dist = outputs["B3"]["out_dist"]
                preds = pred_dist.mean
                if all_preds_dist is None:
                    all_preds_dist = []
                all_preds_dist.append(pred_dist)
            else:
                preds = outputs["B3"]["out"]
            
            all_preds.append(preds.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    if is_probabilistic:
        # Concatenate distributions - we'll extract mean and std separately
        all_means = []
        all_stds = []
        for dist in all_preds_dist:
            all_means.append(dist.mean.cpu().numpy())
            all_stds.append(dist.stddev.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)
        all_stds = np.concatenate(all_stds, axis=0)
        return all_preds, all_means, all_stds
    else:
        return all_preds, None, None


def plot_1D(model, train_dataset, val_dataset, test_dataset, scaler=None, scaler_targets=None, device='cpu', batch_size=32, 
            source_functions=None, use_source_functions=False):
    """
    Plot predictions vs test data for 1D input/output models.
    
    For each source, plots:
    - Test data predictions (with 95% CI if probabilistic)
    - Training data as scatter plot
    - Test targets as line plot (either from test_dataset or generated from source_functions if use_source_functions=True)
    
    Args:
        model: Trained ProNDF model
        train_dataset: MultiFidelityDataset for training data
        val_dataset: MultiFidelityDataset for validation data (not used but kept for API consistency)
        test_dataset: MultiFidelityDataset for test data
        scaler: Optional scaler object with inverse_transform method for unscaling numerical inputs (x-axis).
            If None, inputs are not unscaled.
        scaler_targets: Optional scaler object with inverse_transform method for unscaling targets/predictions (y-axis).
            If None but scaler is provided, scaler will be used for both inputs and targets (backward compatibility).
            If None and scaler is None, targets are not unscaled.
        device: Device to run model on ('cpu' or 'cuda')
        batch_size: Batch size for prediction
        source_functions: Optional list of callable functions, one per source. Each function should take
            (cat, num) as arguments and return targets in unscaled space. Required if use_source_functions=True.
        use_source_functions: If True, generate clean (noiseless) y_true using source_functions instead
            of using test_dataset.targets. Default is False.
    
    Returns:
        matplotlib figure object
    """
    # Get predictions for test data
    test_preds, test_means, test_stds = _get_predictions(model, test_dataset, device, batch_size)
    
    # Validate source_functions if use_source_functions is True
    if use_source_functions:
        if source_functions is None:
            raise ValueError("source_functions must be provided when use_source_functions=True")
        if len(source_functions) != test_dataset.meta['dsource']:
            raise ValueError(f"Number of source_functions ({len(source_functions)}) must match number of sources ({test_dataset.meta['dsource']})")
        if not test_dataset.quant_in:
            raise ValueError("use_source_functions=True requires numerical inputs (quant_in=True) since source functions expect numerical inputs")
    
    # Extract data from datasets
    test_source = test_dataset.source
    test_num = test_dataset.num if test_dataset.quant_in else None
    test_targets = test_dataset.targets
    
    train_source = train_dataset.source
    train_num = train_dataset.num if train_dataset.quant_in else None
    train_targets = train_dataset.targets
    
    # Determine target scaler (use scaler_targets if provided, otherwise fall back to scaler for backward compatibility)
    target_scaler = scaler_targets if scaler_targets is not None else scaler
    
    # Unscale numerical inputs (x-axis) if scaler provided
    if scaler is not None:
        test_num = scaler.inverse_transform(test_num)
        train_num = scaler.inverse_transform(train_num)
    
    # Unscale targets and predictions (y-axis) if target_scaler provided
    if target_scaler is not None:
        # Only unscale test_targets if not using source functions (source functions return unscaled values)
        if not use_source_functions:
            test_targets = target_scaler.inverse_transform(test_targets)
        # Unscale predictions
        test_means = target_scaler.inverse_transform(test_means) if test_means is not None else None
        # Unscale standard deviations (multiply by scale factor)
        # The scale factor is the derivative of the inverse transform
        if test_stds is not None:
            # For StandardNormalizer: inverse_transform is (x * scale_) + mean_
            # Derivative is scale_, so std_unscaled = std_scaled * scale_
            if hasattr(target_scaler, 'with_std'):
                # StandardNormalizer
                test_stds = test_stds * target_scaler.scale_
            elif hasattr(target_scaler, 'denom_'):
                # MinMaxNormalizer: inverse_transform is ((x - min_shift_) / scale_) * denom_ + data_min_
                # Derivative is denom_ / scale_, so std_unscaled = std_scaled * (denom_ / scale_)
                test_stds = test_stds * (target_scaler.denom_ / target_scaler.scale_)
            else:
                # Fallback: try to use scale_ if it exists
                if hasattr(target_scaler, 'scale_'):
                    test_stds = test_stds * target_scaler.scale_
        train_targets = target_scaler.inverse_transform(train_targets)
    
    # Generate clean targets from source functions if requested
    if use_source_functions:
        # Get categorical inputs if present (unscaled if scaler was applied)
        test_cat = test_dataset.cat if test_dataset.qual_in else None
        
        # Generate clean targets for each source
        clean_targets_list = []
        for source_idx in range(test_dataset.meta['dsource']):
            # Get mask for this source
            source_mask = test_source[:, source_idx] == 1
            
            # Get inputs for this source (already unscaled if scaler was provided)
            if test_dataset.qual_in:
                cat_source = test_cat[source_mask, :] if test_cat is not None else None
            else:
                cat_source = None
            
            if test_dataset.quant_in:
                num_source = test_num[source_mask, :] if test_num is not None else None
            else:
                num_source = None
            
            # Call source function to generate clean targets (in unscaled space)
            # Source functions expect (cat, num) and return targets
            # Note: num_source is guaranteed to be not None since we check quant_in above
            clean_targets_source = source_functions[source_idx](cat_source, num_source)
            
            # Convert to numpy array if not already
            if not isinstance(clean_targets_source, np.ndarray):
                clean_targets_source = np.array(clean_targets_source)
            
            # Ensure 2D shape (n_samples, dtargets)
            if clean_targets_source.ndim == 1:
                clean_targets_source = clean_targets_source.reshape(-1, 1)
            elif clean_targets_source.ndim == 0:
                # Scalar result - reshape to (1, 1)
                clean_targets_source = clean_targets_source.reshape(1, 1)
            
            # Store targets for this source (will need to combine later)
            clean_targets_list.append((source_mask, clean_targets_source))
        
        # Combine clean targets in the same order as test_dataset
        test_targets_clean = np.zeros_like(test_targets)
        for source_mask, clean_targets_source in clean_targets_list:
            test_targets_clean[source_mask, :] = clean_targets_source
        
        # Replace test_targets with clean targets
        test_targets = test_targets_clean
    
    # Get number of sources
    n_sources = test_source.shape[1]
    
    # Create figure
    fig, ax = plt.subplots(nrows=n_sources, ncols=1)
    if n_sources == 1:
        ax = [ax]
    fig.set_figheight(7 * n_sources)
    fig.set_figwidth(7)
    
    is_probabilistic = model.B3.probabilistic_output
    
    for i, curr_ax in enumerate(ax):
        # Get data for this source
        source_mask = test_source[:, i] == 1
        test_num_source = test_num[source_mask, :] if test_num is not None else None
        test_targets_source = test_targets[source_mask, :]
        test_preds_source = test_preds[source_mask, :]
        
        train_source_mask = train_source[:, i] == 1
        train_num_source = train_num[train_source_mask, :] if train_num is not None else None
        train_targets_source = train_targets[train_source_mask, :]
        
        # For 1D, we assume single input and output dimension
        if test_num_source is not None:
            x_test = test_num_source.flatten()
            x_train = train_num_source.flatten()
        else:
            # If no numerical input, use indices (not ideal but handles edge case)
            x_test = np.arange(len(test_targets_source))
            x_train = np.arange(len(train_targets_source))
        
        y_test = test_targets_source.flatten()
        y_train = train_targets_source.flatten()
        y_pred = test_preds_source.flatten()
        
        # Sort by x for plotting
        sort_idx = np.argsort(x_test)
        x_test_sorted = x_test[sort_idx]
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Get min/max for x-axis range
        x_min = np.min(x_test)
        x_max = np.max(x_test)
        x_in = np.linspace(x_min, x_max, num=1000)
        
        # Plot 95% CI if probabilistic
        if is_probabilistic and test_means is not None and test_stds is not None:
            test_means_source = test_means[source_mask, :].flatten()
            test_stds_source = test_stds[source_mask, :].flatten()
            test_means_sorted = test_means_source[sort_idx]
            test_stds_sorted = test_stds_source[sort_idx]
            
            y_upp = test_means_sorted + 1.96 * test_stds_sorted
            y_low = test_means_sorted - 1.96 * test_stds_sorted
            
            curr_ax.fill_between(x_test_sorted, y_low, y_upp, alpha=0.3, color='orange', label='95% CI')
            curr_ax.plot(x_test_sorted, test_means_sorted, '--', color='dodgerblue', lw=2, label='Mean Prediction')
        else:
            curr_ax.plot(x_test_sorted, y_pred_sorted, '--', color='dodgerblue', lw=2, label='Prediction')
        
        # Plot true test targets (clean if use_source_functions=True, noisy otherwise)
        y_true_label = 'y_true (clean)' if use_source_functions else 'y_true'
        curr_ax.plot(x_test_sorted, y_test_sorted, '-', color='deepskyblue', alpha=1, lw=2, label=y_true_label)
        
        # Plot training data
        curr_ax.scatter(x_train, y_train, alpha=0.7, color='darkorange', label='Training Data')
        
        curr_ax.set_title(f'Predictions for Source {i}')
        curr_ax.set_xlabel('x')
        curr_ax.set_ylabel('y')
        curr_ax.legend()
    
    plt.tight_layout()
    return fig


def plot_true_pred(model, test_dataset, device='cpu', batch_size=32, colors=("red", "blue"), lw=2, s=30, figsize=7):
    """
    Plot true vs predicted values for each data source.
    
    Args:
        model: Trained ProNDF model
        test_dataset: MultiFidelityDataset for test data
        device: Device to run model on ('cpu' or 'cuda')
        batch_size: Batch size for prediction
        colors: Tuple of (line_color, scatter_color) for the diagonal line and scatter points
        lw: Line width for diagonal line
        s: Scatter point size
        figsize: Base figure size
    
    Returns:
        matplotlib figure object
    """
    # Get predictions
    test_preds, test_means, _ = _get_predictions(model, test_dataset, device, batch_size)
    
    # Use mean predictions if probabilistic, otherwise use direct predictions
    if test_means is not None:
        y_pred = test_means
    else:
        y_pred = test_preds
    
    # Get true targets
    y_true = test_dataset.targets
    source_OH = test_dataset.source
    
    # Get y_min and y_max
    y_min = np.min(np.concatenate((y_true, y_pred), axis=0)[:])
    y_max = np.max(np.concatenate((y_true, y_pred), axis=0)[:])
    yy = np.linspace(y_min, y_max, num=1000)
    
    n_sources = source_OH.shape[1]
    if n_sources == 1:
        fig, curr_ax = plt.subplots(figsize=(figsize, figsize))
        ax_list = [curr_ax]
    else:
        n_rows = int(np.ceil(n_sources / 2))
        fig, ax_list = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(2 * figsize, n_rows * figsize)
        )
        if n_sources == 2:
            ax_list = ax_list.flatten()
    
    for source_idx in range(n_sources):
        if n_sources == 1:
            curr_ax = ax_list[0]
        elif n_sources == 2:
            curr_ax = ax_list[source_idx]
        else:
            row_idx = source_idx // 2
            col_idx = source_idx % 2
            curr_ax = ax_list[row_idx, col_idx]
        
        curr_ax.plot(yy, yy, color=colors[0], linewidth=lw, label="true = pred")
        
        curr_ax.scatter(
            y_true[source_OH[:, source_idx] == 1, :],
            y_pred[source_OH[:, source_idx] == 1, :],
            s=s,
            c=colors[1],
            alpha=0.7,
            label="True vs Pred",
        )
        curr_ax.set_title(f"Source {source_idx}")
        curr_ax.set_xlabel("y true")
        curr_ax.set_ylabel("y pred")
        curr_ax.legend(loc="upper left")
    
    plt.tight_layout()
    return fig


def plot_2D_latent_space(model, block_idx, dcat, device='cpu', figsize=6, num_iterations=100):
    """
    Plot the 2D latent space for either the source block (B1) or categorical block (B2).
    
    Args:
        model: Trained ProNDF model
        block_idx: 0 for source latent space (B1), 1 for categorical latent space (B2)
        dcat: List of number of categories for each categorical input (required for B2)
        device: Device to run model on ('cpu' or 'cuda')
        figsize: Base figure size
        num_iterations: Number of samples to draw for probabilistic latent spaces
    
    Returns:
        matplotlib figure object
    """
    model.eval()
    model = model.to(device)
    
    if block_idx == 0:
        # Source latent space (B1)
        dsource = model.hparams.dsource
        # Create one-hot encoded source vectors for each source
        source_OH = np.eye(dsource)
        source_tensor = torch.tensor(source_OH, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if model.B1.probabilistic_output:
                z_dist = model.B1.predict_distribution(source_tensor)
                prob_LS = True
            else:
                z = model.B1(source_tensor)
                prob_LS = False
        
        # Create single plot for source latent space
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        
        if prob_LS:
            for itr in range(num_iterations):
                z_temp = z_dist.sample()
                z_temp = z_temp.cpu().numpy()
                for i in range(dsource):
                    label = f"Source {i}" if itr == 0 else None
                    ax.scatter(
                        z_temp[i, 0],
                        z_temp[i, 1],
                        color=my_colors[i % len(my_colors)],
                        alpha=0.5,
                        label=label,
                    )
        else:
            z = z.cpu().numpy()
            for i in range(dsource):
                ax.scatter(
                    z[i, 0],
                    z[i, 1],
                    color=my_colors[i % len(my_colors)],
                    label=f"Source {i}",
                )
        
        ax.set_title("Source Latent Space (B1)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.legend()
        
    elif block_idx == 1:
        # Categorical latent space (B2)
        if not model.hparams.qual_in:
            raise ValueError("Model does not have categorical inputs (qual_in=False)")
        
        # Get total number of categorical combinations and levels
        cat_combs = np.prod(dcat)
        cat_sum = np.sum(dcat)
        # Make a matrix of every possible categorical combination
        cat_raw = np.array(list((itertools.product(*[range(L) for L in dcat]))))
        # Initialize dummy matrix (OH encoding)
        cat_OH = np.empty((cat_combs, cat_sum))
        # Do the one-hot encoding
        start_idx = 0
        for idx, level in enumerate(dcat):
            end_idx = start_idx + level
            cat_OH[:, start_idx:end_idx] = to_categorical(
                cat_raw[:, idx], num_classes=level
            )
            start_idx = end_idx
        
        cat_OH_tensor = torch.tensor(cat_OH, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if model.B2.probabilistic_output:
                z_dist = model.B2.predict_distribution(cat_OH_tensor)
                prob_LS = True
            else:
                z = model.B2(cat_OH_tensor)
                prob_LS = False
        
        # Make a plot for each categorical variable
        num_plots = len(dcat)
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            ax_list = [ax]
        else:
            n_rows = int(np.ceil(num_plots / 2))
            fig, ax_list = plt.subplots(
                nrows=n_rows, ncols=2, figsize=(n_rows * figsize, 2 * figsize)
            )
            if num_plots == 2:
                ax_list = ax_list.flatten()
        
        for idx, level in enumerate(dcat):
            if num_plots == 1:
                curr_ax = ax_list[0]
            elif num_plots == 2:
                curr_ax = ax_list[idx]
            else:
                row_idx = idx // 2
                col_idx = idx % 2
                curr_ax = ax_list[row_idx, col_idx]
            
            curr_ax.set_title(f"Graph of cat{idx + 1}")
            
            for i in range(level):
                if prob_LS:
                    for itr in range(num_iterations):
                        z_temp = z_dist.sample()
                        z_temp = z_temp.cpu().numpy()
                        label = f"level {i + 1}" if itr == 0 else None
                        curr_ax.scatter(
                            z_temp[cat_raw[:, idx] == i, 0],
                            z_temp[cat_raw[:, idx] == i, 1],
                            color=my_colors[i % len(my_colors)],
                            alpha=0.5,
                            label=label,
                        )
                else:
                    z_np = z.cpu().numpy()
                    curr_ax.scatter(
                        z_np[cat_raw[:, idx] == i, 0],
                        z_np[cat_raw[:, idx] == i, 1],
                        color=my_colors[i % len(my_colors)],
                        label=f"level {i + 1}",
                    )
            curr_ax.legend()
    else:
        raise ValueError("block_idx must be 0 (B1/source) or 1 (B2/categorical)")
    
    plt.tight_layout()
    return fig


