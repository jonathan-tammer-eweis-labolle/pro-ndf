"""
Generate and save the rational function dataset splits.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from prondf import data

# Set save path
save_path = "datasets/rational/generated_data/"

# Set seed
np.random.seed(42)

# Set number of samples
num_samples = [800, 4800, 4800, 4800]

# Set source functions
def yh(cat, x):
    """High-fidelity rational response function."""
    return 1.0 / ((0.1 * x[:] ** 3) + (x[:] ** 2) + x[:] + 1)

def yl1(cat, x):
    """Low-fidelity rational response function (variant 1)."""
    return 1.0 / ((0.2 * x[:] ** 3) + (x[:] ** 2) + x[:] + 1)

def yl2(cat, x):
    """Low-fidelity rational response function (variant 2)."""
    return 1.0 / ((x[:] ** 2) + x[:] + 1)

def yl3(cat, x):
    """Low-fidelity rational response function (variant 3)."""
    return 1.0 / ((x[:] ** 2) + 1)
source_functions = [yh, yl1, yl2, yl3]

# Make dataset
dataset = data.Generate_Analytic_Dataset(
    dsource = 4,
    dcat = None,
    dnum = 1,
    dtargets = 1,
    qual_in = False,
    quant_in = True,
    num_samples = num_samples,
    source_functions = source_functions,
    num_ranges = [(-2.0, 3.0)],
    noise_variance = [(0.001,), (0.001,), (0.001,), (0.001,)],
    random_generator = np.random.default_rng(42),
)

# Save dataset
dataset.save(save_path, "rational_dataset")

# Load dataset
dataset = data.MultiFidelityDataset.load(save_path, "rational_dataset")

# Split dataset
split_ratios = [5/800, 3/800, 0.99]
train_dataset, val_dataset, test_dataset = data.split_dataset(dataset, split_ratios)

# Save datasets
data.save_splits(train_dataset, val_dataset, test_dataset, save_path, "rational_dataset")