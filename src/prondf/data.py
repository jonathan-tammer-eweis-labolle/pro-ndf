"""
data.py
This module contains classes and functions for generating and handling datasets to be
used with ProNDF. User may add additional or custom datasets via files, which can then
be loaded.
"""

from __future__ import annotations
import torch
import numpy as np
from scipy.stats import qmc
from torch.utils.data import Dataset, DataLoader, Sampler
from dataclasses import dataclass, field
from typing import Optional, Tuple, TypedDict
import json
import os
import warnings
from pathlib import Path


# Dataset metadata type definition
class DatasetMeta(TypedDict, total=False):
    """
    Metadata dictionary for MultiFidelityDataset.
    
    All fields are optional (total=False), allowing users to provide only the
    metadata they need. Currently used fields:
    - quant_in: Whether quantitative (numerical) inputs are present.
    - qual_in: Whether qualitative (categorical) inputs are present.
    - dsource: Number of data sources.
    - dcat: Number of categories for each categorical input.
    - dnum: Dimension of the numerical input.
    - dtargets: Dimension of the targets / number of output dimensions.
    - num_samples: Number of samples to generate for each source.
    """
    dsource: int
    dcat: list[int]
    dnum: int
    dtargets: int
    qual_in: bool
    quant_in: bool
    num_samples: list[int]

# Main dataset class

class MultiFidelityDataset(Dataset):
    """
    PyTorch Dataset for multi-fidelity data with optional categorical and numerical inputs.
    
    This dataset stores one-hot source indicators, optional categorical inputs, optional
    numerical inputs, and target outputs, along with metadata describing dimensions.
    
    Args:
        source: Array of one-hot encoded source indicators with shape (n_samples, dsource).
        cat: Optional categorical inputs with shape (n_samples, sum(dcat)).
        num: Optional numerical inputs with shape (n_samples, dnum).
        targets: Target outputs with shape (n_samples, dtargets).
        meta: Metadata dictionary describing dataset dimensions and properties.
    """
    def __init__(
            self,
            source: np.ndarray = None,
            cat: np.ndarray = None,
            num: np.ndarray = None,
            targets: np.ndarray = None,
            meta: DatasetMeta = {},
    ):
        super(MultiFidelityDataset, self).__init__()
        # Store parameters
        self.source = source
        self.cat = cat
        self.num = num
        self.targets = targets
        self.meta = meta
        # Unpack meta (NOTE: Update at a later date if needed - may need more / fewer params)
        self.quant_in = meta.get('quant_in', None)
        self.qual_in = meta.get('qual_in', None)

    def __len__(self):
        """Returns number of samples in the dataset."""
        return self.targets.shape[0]  # Targets are always present in the dataset
        
    def __getitem__(self, idx):
        """
        Returns a dict containing a single sample from the dataset.
        Args:
            idx (int): index of the sample to retrieve.
        Returns:
            sample: a dict containing the sample data.
            Keys: source, cat, num, targets
        """
        source_sample = torch.tensor(self.source[idx, :], dtype=torch.float32)
        targets_sample = torch.tensor(self.targets[idx, :], dtype=torch.float32)
        # Retrieve samples from categorical and numerical inputs if they exist
        # When qual_in/quant_in are False, create empty tensors with 0 features (not same shape as targets)
        if self.qual_in and self.cat is not None:
            cat_sample = torch.tensor(self.cat[idx, :], dtype=torch.float32)
        else:
            # Create empty tensor with 0 features (shape: (0,))
            cat_sample = torch.empty(0, dtype=torch.float32)
        if self.quant_in and self.num is not None:
            num_sample = torch.tensor(self.num[idx, :], dtype=torch.float32)
        else:
            # Create empty tensor with 0 features (shape: (0,))
            num_sample = torch.empty(0, dtype=torch.float32)
        out = {
            'source': source_sample,
            'cat': cat_sample,
            'num': num_sample,
            'targets': targets_sample
        }
        return out
    
    def save(self, folder_path: str | Path, filename: str):
        """
        Saves the dataset to disk.
        
        Saves each data array as a separate .npy file and metadata as a JSON file.
        Files are saved as:
        - {filename}_source.npy
        - {filename}_cat.npy (if qual_in is True)
        - {filename}_num.npy (if quant_in is True)
        - {filename}_targets.npy
        - {filename}_meta.json
        
        Args:
            folder_path: Path to the folder where the dataset should be saved.
            filename: Base filename (without extension) for the saved files.
        
        Raises:
            OSError: If the folder cannot be created or files cannot be written.
        """
        # Convert to Path object and create folder if it doesn't exist
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        np.save(folder_path / f"{filename}_source.npy", self.source)
        np.save(folder_path / f"{filename}_targets.npy", self.targets)
        
        # Save optional arrays if they exist (check both flag and array presence for safety)
        if self.cat is not None:
            np.save(folder_path / f"{filename}_cat.npy", self.cat)
        
        if self.num is not None:
            np.save(folder_path / f"{filename}_num.npy", self.num)
        
        # Save metadata as JSON (convert numpy types to Python types for JSON serialization)
        def convert_to_serializable(obj):
            """Recursively convert numpy types and lists to JSON-serializable types."""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_meta = convert_to_serializable(self.meta)
        with open(folder_path / f"{filename}_meta.json", 'w') as f:
            json.dump(serializable_meta, f, indent=2)
    
    @classmethod
    def load(cls, folder_path: str | Path, filename: str) -> "MultiFidelityDataset":
        """
        Loads a dataset from disk.
        
        Loads arrays and metadata that were saved using the save() method.
        
        Args:
            folder_path: Path to the folder containing the dataset files.
            filename: Base filename (without extension) used when saving.
        
        Returns:
            A MultiFidelityDataset object with the loaded data and metadata.
        
        Raises:
            FileNotFoundError: If required files are not found.
            ValueError: If metadata cannot be loaded or is invalid.
        """
        folder_path = Path(folder_path)
        
        # Check that required files exist
        source_file = folder_path / f"{filename}_source.npy"
        targets_file = folder_path / f"{filename}_targets.npy"
        meta_file = folder_path / f"{filename}_meta.json"
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source data file not found: {source_file}")
        if not targets_file.exists():
            raise FileNotFoundError(f"Targets data file not found: {targets_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        
        # Load arrays
        source = np.load(source_file)
        targets = np.load(targets_file)
        
        # Load metadata
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Load optional arrays if they exist
        cat = None
        num = None
        qual_in = meta.get('qual_in', False)
        quant_in = meta.get('quant_in', False)
        
        cat_file = folder_path / f"{filename}_cat.npy"
        num_file = folder_path / f"{filename}_num.npy"
        
        # Load arrays if files exist (check both file existence and metadata flag)
        if cat_file.exists():
            cat = np.load(cat_file)
            # Warn if metadata doesn't indicate categorical data should be present
            if not qual_in:
                warnings.warn(
                    f"Categorical data file found ({cat_file}) but metadata indicates "
                    "qual_in=False. Loading categorical data anyway."
                )
        elif qual_in:
            # Metadata says it should exist but file doesn't
            raise FileNotFoundError(
                f"Categorical data file not found: {cat_file}. "
                "Metadata indicates categorical data should be present."
            )
        
        if num_file.exists():
            num = np.load(num_file)
            # Warn if metadata doesn't indicate numerical data should be present
            if not quant_in:
                warnings.warn(
                    f"Numerical data file found ({num_file}) but metadata indicates "
                    "quant_in=False. Loading numerical data anyway."
                )
        elif quant_in:
            # Metadata says it should exist but file doesn't
            raise FileNotFoundError(
                f"Numerical data file not found: {num_file}. "
                "Metadata indicates numerical data should be present."
            )
        
        # Create and return dataset
        return cls(
            source=source,
            cat=cat,
            num=num,
            targets=targets,
            meta=meta
        )


# Function to split a dataset into train, validation, and test sets
def split_dataset(
    dataset: MultiFidelityDataset,
    split_ratios: list[float],
    random_generator: np.random.Generator = None,
) -> tuple[MultiFidelityDataset, MultiFidelityDataset, MultiFidelityDataset]:
    """
    Splits a dataset into train, validation, and test sets.

    The split is performed separately for each data source, ensuring that each source
    is split according to the specified ratios. This maintains the distribution of
    sources across the splits.

    Args:
        dataset: The MultiFidelityDataset to split.
        split_ratios: A list of 3 floats [train_ratio, val_ratio, test_ratio] that
            must sum to 1.0. For example, [0.7, 0.2, 0.1] for 70% train, 20% val, 10% test.
        random_generator: Optional numpy Generator used to shuffle each source's samples
            before splitting. Pass a seeded generator (e.g. np.random.default_rng(0)) for
            reproducible splits. If None, a fresh, unseeded generator is created.

    Returns:
        A tuple of three MultiFidelityDataset objects: (train_dataset, val_dataset, test_dataset).
        Each dataset has updated metadata with num_samples reflecting the split for each source.

    Example:
        >>> train, val, test = split_dataset(dataset, [0.7, 0.2, 0.1])
    """
    # Sanity checks
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must be a list of 3 floats")
    if abs(sum(split_ratios) - 1.0) > 1e-6:  # Use small epsilon for floating point comparison
        raise ValueError("split_ratios must sum to 1.0")
    # Initialize random generator if not provided (seed it for reproducible splits)
    if random_generator is None:
        random_generator = np.random.default_rng()

    # Get number of sources from metadata
    dsource = dataset.meta.get('dsource')
    if dsource is None:
        raise ValueError("Dataset metadata must contain 'dsource'")
    
    # Initialize lists to collect indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Track num_samples for each split and source
    train_num_samples = []
    val_num_samples = []
    test_num_samples = []
    
    # Split each source separately
    for source_idx in range(dsource):
        # Find indices where this source is active (one-hot encoded)
        source_mask = dataset.source[:, source_idx] == 1
        source_indices = np.where(source_mask)[0]
        num_source_samples = len(source_indices)
        
        if num_source_samples == 0:
            # No samples for this source
            train_num_samples.append(0)
            val_num_samples.append(0)
            test_num_samples.append(0)
            continue
        
        # Shuffle indices for randomness (use the provided generator for reproducibility)
        random_generator.shuffle(source_indices)
        
        # Calculate split sizes for this source
        train_size_source = int(num_source_samples * split_ratios[0])
        val_size_source = int(num_source_samples * split_ratios[1])
        test_size_source = num_source_samples - train_size_source - val_size_source
        
        # Split indices for this source
        train_indices_source = source_indices[:train_size_source]
        val_indices_source = source_indices[train_size_source:train_size_source + val_size_source]
        test_indices_source = source_indices[train_size_source + val_size_source:]
        
        # Collect indices and counts
        train_indices.extend(train_indices_source)
        val_indices.extend(val_indices_source)
        test_indices.extend(test_indices_source)
        
        train_num_samples.append(len(train_indices_source))
        val_num_samples.append(len(val_indices_source))
        test_num_samples.append(len(test_indices_source))
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Split the data arrays
    train_source = dataset.source[train_indices, :]
    train_cat = dataset.cat[train_indices, :] if dataset.qual_in else None
    train_num = dataset.num[train_indices, :] if dataset.quant_in else None
    train_targets = dataset.targets[train_indices, :]
    
    val_source = dataset.source[val_indices, :]
    val_cat = dataset.cat[val_indices, :] if dataset.qual_in else None
    val_num = dataset.num[val_indices, :] if dataset.quant_in else None
    val_targets = dataset.targets[val_indices, :]
    
    test_source = dataset.source[test_indices, :]
    test_cat = dataset.cat[test_indices, :] if dataset.qual_in else None
    test_num = dataset.num[test_indices, :] if dataset.quant_in else None
    test_targets = dataset.targets[test_indices, :]
    
    # Create new metadata for each split (copy and update num_samples)
    train_meta = dataset.meta.copy()
    train_meta['num_samples'] = train_num_samples
    
    val_meta = dataset.meta.copy()
    val_meta['num_samples'] = val_num_samples
    
    test_meta = dataset.meta.copy()
    test_meta['num_samples'] = test_num_samples
    
    # Create new dataset objects
    train_dataset = MultiFidelityDataset(
        source=train_source,
        cat=train_cat,
        num=train_num,
        targets=train_targets,
        meta=train_meta
    )
    
    val_dataset = MultiFidelityDataset(
        source=val_source,
        cat=val_cat,
        num=val_num,
        targets=val_targets,
        meta=val_meta
    )
    
    test_dataset = MultiFidelityDataset(
        source=test_source,
        cat=test_cat,
        num=test_num,
        targets=test_targets,
        meta=test_meta
    )
    
    return train_dataset, val_dataset, test_dataset


def save_splits(
    train_dataset: MultiFidelityDataset,
    val_dataset: MultiFidelityDataset,
    test_dataset: MultiFidelityDataset,
    folder_path: str | Path,
    filename: str
) -> None:
    """
    Saves train, validation, and test dataset splits to disk.
    
    Each split is saved with a suffix (_train, _val, _test) appended to the base filename.
    This function is a convenience wrapper that calls save() on each dataset split.
    
    Args:
        train_dataset: The training dataset split.
        val_dataset: The validation dataset split.
        test_dataset: The test dataset split.
        folder_path: Path to the folder where the splits should be saved.
        filename: Base filename (without extension) for the saved files. Each split will
            have _train, _val, or _test appended to this base name.
    
    Example:
        >>> train, val, test = split_dataset(dataset, [0.7, 0.2, 0.1])
        >>> save_splits(train, val, test, "./data", "my_dataset")
        # Saves:
        # ./data/my_dataset_train_source.npy, ./data/my_dataset_train_targets.npy, etc.
        # ./data/my_dataset_val_source.npy, ./data/my_dataset_val_targets.npy, etc.
        # ./data/my_dataset_test_source.npy, ./data/my_dataset_test_targets.npy, etc.
    """
    train_dataset.save(folder_path, f"{filename}_train")
    val_dataset.save(folder_path, f"{filename}_val")
    test_dataset.save(folder_path, f"{filename}_test")


def load_splits(
    folder_path: str | Path,
    filename: str
) -> tuple[MultiFidelityDataset, MultiFidelityDataset, MultiFidelityDataset]:
    """
    Loads train, validation, and test dataset splits from disk.
    
    Each split is loaded using a suffix (_train, _val, _test) appended to the base filename.
    This function is a convenience wrapper that calls load() on each dataset split.
    
    Args:
        folder_path: Path to the folder containing the dataset split files.
        filename: Base filename (without extension) used when saving. Each split will
            have _train, _val, or _test appended to this base name.
    
    Returns:
        A tuple of three MultiFidelityDataset objects: (train_dataset, val_dataset, test_dataset).
    
    Raises:
        FileNotFoundError: If any required files are not found.
        ValueError: If metadata cannot be loaded or is invalid.
    
    Example:
        >>> train, val, test = load_splits("./data", "my_dataset")
        # Loads:
        # ./data/my_dataset_train_source.npy, ./data/my_dataset_train_targets.npy, etc.
        # ./data/my_dataset_val_source.npy, ./data/my_dataset_val_targets.npy, etc.
        # ./data/my_dataset_test_source.npy, ./data/my_dataset_test_targets.npy, etc.
    """
    train_dataset = MultiFidelityDataset.load(folder_path, f"{filename}_train")
    val_dataset = MultiFidelityDataset.load(folder_path, f"{filename}_val")
    test_dataset = MultiFidelityDataset.load(folder_path, f"{filename}_test")
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    """
    Custom collate function for MultiFidelityDataset to stack dictionary samples into batched format.
    
    This function takes a list of sample dictionaries (one per sample) and converts them into
    a single batched dictionary with stacked tensors. This is required when using PyTorch's
    DataLoader with MultiFidelityDataset, which returns dictionaries.
    
    Args:
        batch: List of sample dictionaries, where each dictionary has keys:
            - 'source': tensor of shape (dsource,)
            - 'cat': tensor of shape (sum(dcat),) or empty tensor
            - 'num': tensor of shape (dnum,) or empty tensor
            - 'targets': tensor of shape (dtargets,)
    
    Returns:
        Dictionary with batched tensors:
            - 'source': tensor of shape (batch_size, dsource)
            - 'cat': tensor of shape (batch_size, sum(dcat)) or (batch_size, 0)
            - 'num': tensor of shape (batch_size, dnum) or (batch_size, 0)
            - 'targets': tensor of shape (batch_size, dtargets)
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> from data import MultiFidelityDataset, collate_fn
        >>> 
        >>> dataset = MultiFidelityDataset(...)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    source_list = []
    cat_list = []
    num_list = []
    targets_list = []
    
    for sample in batch:
        source_list.append(sample['source'])
        cat_list.append(sample['cat'])
        num_list.append(sample['num'])
        targets_list.append(sample['targets'])
    
    return {
        'source': torch.stack(source_list),
        'cat': torch.stack(cat_list),
        'num': torch.stack(num_list),
        'targets': torch.stack(targets_list)
    }


# Functions and classes dealing with normalizing/scaling or otherwise preprocessing data

def to_categorical(x, num_classes=None):
    """
    The "to_categorical" function copied directly from the TensorFlow documentation.
    NOT my own code, but included here to avoid TensorFlow dependency.

    Converts a class vector (integers) to binary class matrix (i.e., one-hot encoding).

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

ArrayLike = np.ndarray  # For type hinting

def _ensure_2d(array: ArrayLike) -> Tuple[ArrayLike, bool]:
    """
    Ensures that input array is 2D. If input is 1D, reshapes to (N, 1).
    """
    x = np.asarray(array)
    was_1d = (x.ndim == 1)
    if was_1d:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Input array must be 1D or 2D array-like.")
    return x, was_1d

@dataclass
class StandardNormalizer:
    """
    Standardizes features (column-wise) to zero mean and unit variance:
    z = (x - mean) / std
    Stores mean_ and scale_ for inverse transform.
    Zero-variance columns have their scale_ set to 1 to avoid division by zero.
    """
    with_mean: bool = True
    with_std: bool = True
    mean_: Optional[np.ndarray] = field(init=False, default=None)
    scale_: Optional[np.ndarray] = field(init=False, default=None)
    n_features_in_: Optional[int] = field(init=False, default=None)

    def fit(self, x: ArrayLike) -> "StandardNormalizer":
        """
        Stores parameters for normalizing / unnormalizing data.
        """
        x2d, _ = _ensure_2d(x)
        self.n_features_in_ = x2d.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(x2d, axis=0)
        else:
            self.mean_ = np.zeros(x2d.shape[1], dtype=x2d.dtype)
        if self.with_std:
            std = np.nanstd(x2d, axis=0, ddof=0)  # ddof=0 to match conventions
            # Handle zero-variance edge cases
            std_safe = std.copy()
            std_safe[std_safe == 0] = 1.0
            self.scale_ = std_safe
        else:
            self.scale_ = np.ones(x2d.shape[1], dtype=x2d.dtype)
        return self
    
    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Normalizes input using the stored parameters.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")
        x2d, was_1d = _ensure_2d(x)
        x2d = x2d - self.mean_ if self.with_mean else x2d
        x2d = x2d / self.scale_ if self.with_std else x2d
        return x2d.ravel() if was_1d else x2d  # Return to 1D if input was 1D
    
    def inverse_transform(self, x: ArrayLike) -> ArrayLike:
        """
        Un-normalizes input using the stored parameters.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        x2d, was_1d = _ensure_2d(x)
        x2d = x2d * self.scale_ if self.with_std else x2d
        x2d = x2d + self.mean_ if self.with_mean else x2d
        return x2d.ravel() if was_1d else x2d  # Return to 1D if input was 1D
    
    def to_dict(self) -> dict:
        """
        Stores parameters in dictionary for serialization.
        """
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardNormalizer":
        """
        Loads parameters from dictionary for deserialization.
        """
        obj = cls(with_mean=d["with_mean"], with_std=d["with_std"])
        obj.mean_ = None if d["mean_"] is None else np.array(d["mean_"])
        obj.scale_ = None if d["scale_"] is None else np.array(d["scale_"])
        obj.n_features_in_ = d.get("n_features_in_", None)
        return obj
    

@dataclass
class MinMaxNormalizer:
    """
    Scales features (column-wise) to provided range (defaults to (0, 1)):
    z = (x - min) / (max - min)
    Zero-variance columns are mapped to the lower bound of the range.
    """
    feature_range: Tuple[float, float] = (0.0, 1.0)
    data_min_: Optional[np.ndarray] = field(init=False, default=None)
    data_max_: Optional[np.ndarray] = field(init=False, default=None)
    scale_: Optional[np.ndarray] = field(init=False, default=None)  # for convenience for inverse transform
    min_shift_: Optional[np.ndarray] = field(init=False, default=None) # for convenience for inverse transform
    denom_: Optional[np.ndarray] = field(init=False, default=None)  # For handling zero-variance columns
    n_features_in_: Optional[int] = field(init=False, default=None)

    def fit(self, x: ArrayLike) -> "MinMaxNormalizer":
        """
        Stores parameters for normalizing / unnormalizing data.
        """
        x2d, _ = _ensure_2d(x)
        self.n_features_in_ = x2d.shape[1]
        self.data_min_ = np.nanmin(x2d, axis=0)
        self.data_max_ = np.nanmax(x2d, axis=0)
        denom = (self.data_max_ - self.data_min_).astype(float)
        denom[denom == 0] = 1.0  # avoid div-by-zero; constant columns handled below
        self.denom_ = denom

        fr_min, fr_max = self.feature_range
        fr_scale = (fr_max - fr_min)
        self.scale_ = np.full(x2d.shape[1], fr_scale, dtype=float)
        self.min_shift_ = np.full(x2d.shape[1], fr_min, dtype=float)
        return self
    
    def transform(self, x: ArrayLike) -> ArrayLike:
        """
        Normalizes input using the stored parameters.
        """
        if any(v is None for v in (self.data_min_, self.denom_, self.scale_, self.min_shift_)):
            raise RuntimeError("Call fit() before transform().")
        x2d, was_1d = _ensure_2d(x)
        z = (x2d - self.data_min_) / self.denom_
        out = z * self.scale_ + self.min_shift_
        # For truly constant columns, map everything to lower bound:
        const_mask = (self.data_max_ == self.data_min_)
        if np.any(const_mask):
            out[:, const_mask] = self.feature_range[0]
        return out.ravel() if was_1d else out  # Return to 1D if input was 1D
    
    def inverse_transform(self, x: ArrayLike) -> ArrayLike:
        """
        Un-normalizes input using the stored parameters.
        """
        if any(v is None for v in (self.data_min_, self.denom_, self.scale_, self.min_shift_)):
            raise RuntimeError("Call fit() before inverse_transform().")
        x2d, was_1d = _ensure_2d(x)
        # Undo feature_range mapping:
        z = (x2d - self.min_shift_) / self.scale_
        out = z * self.denom_ + self.data_min_
        # For constant columns, everything maps back to the constant value:
        const_mask = (self.data_max_ == self.data_min_)
        if np.any(const_mask):
            out[:, const_mask] = self.data_min_[const_mask]
        return out.ravel() if was_1d else out  # Return to 1D if input was 1D
    
    def to_dict(self) -> dict:
        """
        Stores parameters in dictionary for serialization.
        """
        return {
            "feature_range": self.feature_range,
            "data_min_": None if self.data_min_ is None else self.data_min_.tolist(),
            "data_max_": None if self.data_max_ is None else self.data_max_.tolist(),
            "denom_": None if self.denom_ is None else self.denom_.tolist(),
            "scale_": None if self.scale_ is None else self.scale_.tolist(),
            "min_shift_": None if self.min_shift_ is None else self.min_shift_.tolist(),
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxNormalizer":
        """
        Loads parameters from dictionary for deserialization.
        """
        obj = cls(feature_range=tuple(d["feature_range"]))
        obj.data_min_ = None if d["data_min_"] is None else np.array(d["data_min_"])
        obj.data_max_ = None if d["data_max_"] is None else np.array(d["data_max_"])
        obj.denom_ = None if d["denom_"] is None else np.array(d["denom_"])
        obj.scale_ = None if d["scale_"] is None else np.array(d["scale_"])
        obj.min_shift_ = None if d["min_shift_"] is None else np.array(d["min_shift_"])
        obj.n_features_in_ = d.get("n_features_in_", None)
        return obj
    

# Function to generate analytic datasets
def Generate_Analytic_Dataset(
    dsource: int,
    dcat: list[int],
    dnum: int,
    dtargets: int,
    qual_in: bool,
    quant_in: bool,
    num_samples: list[int],
    source_functions: list[callable],
    num_ranges: list[tuple[float, float]] = None,
    noise_variance: list[tuple[float, float, ...]] = None,
    random_generator: np.random.Generator = None,
):
    """
    Generates an analytic dataset with the specified parameters.
    Args:
        dsource: Number of data sources.
        dcat: Number of categories for each categorical input.
        dnum: Dimension of the numerical input.
        dtargets: Dimension of the targets / number of output dimensions.
        qual_in: Whether qualitative (categorical) inputs are present.
        quant_in: Whether quantitative (numerical) inputs are present.
        num_samples: Number of samples to generate for each source.
        source_functions: The functional forms of each data source. Should take a 
            categorical and numerical input and return an array of shape 
            (num_samples, dtargets). Refer to example notebooks for more details.
        num_ranges: The ranges for each numerical input. Should be a list of tuples of 
            length dnum, where each tuple contains the minimum and maximum possible
            values for the corresponding numerical dimension.
        noise_variance: Noise variance for each output dimension for each source. 
            Should be a list of length dsource containing tuples of length dtargets, where 
            each tuple contains the desired noise variances for each output dimension.
        random_generator: Random generator to use for generating the dataset.
    Returns:
        A MultiFidelityDataset object containing the generated dataset.
    """
    # Sanity checks
    if quant_in and num_ranges is None:
        raise ValueError("num_ranges must be provided if quant_in is True")
    if len(num_samples) != dsource:
        raise ValueError("num_samples must be provided for each source")
    if len(source_functions) != dsource:
        raise ValueError("source_functions must be provided for each source")
    if noise_variance is None:
        raise ValueError("noise_variance must be provided for each source")
    if len(noise_variance) != dsource:
        raise ValueError("noise_variance must be provided for each source")
    if len(noise_variance[0]) != dtargets:
        raise ValueError("noise_variance must be provided for each output dimension")
    if qual_in and (dcat is None or len(dcat) == 0):
        raise ValueError("dcat must be a non-empty list when qual_in=True")
    if quant_in and len(num_ranges) != dnum:
        raise ValueError("The number of numerical input ranges must be the same as the number of numerical inputs")
    # Initialize random generator if not provided
    if random_generator is None:
        random_generator = np.random.default_rng()
    # Compute Sobol sampler dimensionality and per-type ranges/offsets
    # Dimension order in Sobol samples: [num dims (if quant_in), cat dims (if qual_in)]
    d_all = 0
    num_col_start = None
    cat_col_start = None
    if quant_in:
        num_col_start = d_all
        d_all += dnum
        num_min = np.array([num_ranges[i][0] for i in range(dnum)])
        num_max = np.array([num_ranges[i][1] for i in range(dnum)])
        num_range_width = num_max - num_min
    if qual_in:
        cat_col_start = d_all
        d_all += len(dcat)
        cat_min = np.zeros(len(dcat))
        cat_range_width = np.array(dcat, dtype=float)
    sobol_sampler = qmc.Sobol(d_all, scramble=True, seed=random_generator)
    # Generate data for each source
    source_data = []
    cat_data = []
    num_data = []
    targets_data = []
    for i in range(dsource):
        num_samples_temp = num_samples[i]
        # Set source inputs
        source_temp = np.zeros((num_samples_temp, dsource))
        source_temp[:, i] = 1  # One-hot encoding
        # Draw one batch of Sobol samples covering all input dimensions
        sobol_samples = sobol_sampler.random(num_samples_temp)
        # Extract and scale numerical inputs
        if quant_in:
            num_temp = sobol_samples[:, num_col_start:num_col_start + dnum] * num_range_width + num_min
        else:
            num_temp = None
        # Extract and scale categorical inputs (floor to integer levels)
        if qual_in:
            cat_temp = np.floor(
                sobol_samples[:, cat_col_start:cat_col_start + len(dcat)] * cat_range_width + cat_min
            )
        else:
            cat_temp = None
        # Generate targets
        targets_temp = source_functions[i](cat_temp, num_temp)
        # Ensure targets are 2D (n_samples, dtargets)
        targets_temp, _ = _ensure_2d(targets_temp)
        # Add noise to targets if present
        if noise_variance[i] is not None:
            targets_temp += random_generator.normal(0, np.sqrt(noise_variance[i]), size=targets_temp.shape)
        # Append to data lists
        source_data.append(source_temp)
        cat_data.append(cat_temp)
        num_data.append(num_temp)
        targets_data.append(targets_temp)
    # Combine data lists into arrays
    source_data = np.concatenate(source_data, axis=0)
    if qual_in:
        cat_data = np.concatenate(cat_data, axis=0)
    else:
        cat_data = None
    if quant_in:
        num_data = np.concatenate(num_data, axis=0)
    else:
        num_data = None
    targets_data = np.concatenate(targets_data, axis=0)
    # Build metadata
    meta = DatasetMeta(
        dsource=dsource,
        dcat=dcat,
        dnum=dnum,
        dtargets=dtargets,
        qual_in=qual_in,
        quant_in=quant_in,
        num_samples=num_samples,
    )
    # Return dataset
    return MultiFidelityDataset(source_data, cat_data, num_data, targets_data, meta)


class StratifiedSourceSampler(Sampler):
    """
    Sampler that ensures each batch contains at least one sample from each source.
    This prevents empty source splits that can cause NaN losses or inconsistent loss values.
    
    The sampler groups indices by source and ensures balanced sampling across sources.
    """
    def __init__(self, dataset: MultiFidelityDataset, batch_size: int, shuffle: bool = True, generator=None):
        """
        Initializes the stratified sampler.
        Args:
            dataset (MultiFidelityDataset): The dataset to sample from.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the indices. Default is True.
            generator (torch.Generator, optional): Random number generator for reproducibility.
        """
        if not isinstance(dataset, MultiFidelityDataset):
            raise TypeError("StratifiedSourceSampler requires a MultiFidelityDataset")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        
        # Get number of sources from metadata
        if 'dsource' not in dataset.meta:
            raise ValueError("Dataset metadata must contain 'dsource'")
        self.num_sources = dataset.meta['dsource']
        
        # Group indices by source
        self.source_indices = [[] for _ in range(self.num_sources)]
        for idx in range(len(dataset)):
            # Get source for this sample (one-hot encoded)
            source = dataset.source[idx]
            source_idx = np.argmax(source)  # Find which source this sample belongs to
            self.source_indices[source_idx].append(idx)
        
        # Check that we have at least one sample per source
        min_samples_per_source = min(len(indices) for indices in self.source_indices)
        if min_samples_per_source == 0:
            warnings.warn(
                "Some sources have no samples. Stratified sampling may not work correctly.",
                UserWarning
            )
        
        # Calculate number of batches
        # Each batch needs at least one sample from each source
        # Remaining samples can be distributed across sources
        min_batch_size = self.num_sources
        if batch_size < min_batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) must be at least {min_batch_size} "
                f"(number of sources) for stratified sampling."
            )
        
        # Calculate how many batches we can make
        # Strategy: ensure each batch has at least one from each source,
        # then fill remaining slots proportionally
        self.num_batches = len(dataset) // batch_size
    
    def __iter__(self):
        """
        Returns an iterator over batch indices, ensuring each batch has samples from all sources.
        """
        # Create a list of batches
        batches = []
        
        # Create shuffled copies of source indices
        source_indices_shuffled = []
        for source_idx_list in self.source_indices:
            indices = source_idx_list.copy()
            if self.shuffle:
                if self.generator is not None:
                    # Use generator for reproducibility
                    indices = indices.copy()
                    for i in range(len(indices) - 1, 0, -1):
                        j = int(torch.randint(0, i + 1, (1,), generator=self.generator).item())
                        indices[i], indices[j] = indices[j], indices[i]
                else:
                    np.random.shuffle(indices)
            source_indices_shuffled.append(indices)
        
        # Create batches ensuring at least one sample from each source
        batch_idx = 0
        source_positions = [0] * self.num_sources  # Track position in each source's index list
        
        while batch_idx < self.num_batches:
            batch = []
            
            # First, add one sample from each source
            for source_idx in range(self.num_sources):
                if source_positions[source_idx] < len(source_indices_shuffled[source_idx]):
                    batch.append(source_indices_shuffled[source_idx][source_positions[source_idx]])
                    source_positions[source_idx] += 1
            
            # Fill remaining slots in batch proportionally across sources
            remaining_slots = self.batch_size - len(batch)
            total_remaining = sum(
                len(source_indices_shuffled[i]) - source_positions[i] 
                for i in range(self.num_sources)
            )
            
            if total_remaining > 0:
                # Distribute remaining slots proportionally
                for _ in range(remaining_slots):
                    # Find source with most remaining samples (proportional distribution)
                    best_source = None
                    best_ratio = -1
                    for source_idx in range(self.num_sources):
                        remaining = len(source_indices_shuffled[source_idx]) - source_positions[source_idx]
                        if remaining > 0:
                            # Ratio of remaining samples for this source
                            ratio = remaining / total_remaining
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_source = source_idx
                    
                    if best_source is not None:
                        batch.append(source_indices_shuffled[best_source][source_positions[best_source]])
                        source_positions[best_source] += 1
                        total_remaining -= 1
                    else:
                        break  # No more samples available
            
            if len(batch) > 0:
                batches.append(batch)
                batch_idx += 1
            else:
                break  # No more samples to create batches
        
        # Shuffle batches if requested
        if self.shuffle and self.generator is None:
            np.random.shuffle(batches)
        elif self.shuffle and self.generator is not None:
            # Shuffle using generator
            for i in range(len(batches) - 1, 0, -1):
                j = int(torch.randint(0, i + 1, (1,), generator=self.generator).item())
                batches[i], batches[j] = batches[j], batches[i]
        
        # Flatten batches into single list of indices
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        """Returns the number of samples that will be yielded."""
        return self.num_batches * self.batch_size
