"""
Generate and save the wing weight dataset splits.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from prondf import data

# Set save path
save_path = "datasets/wing_weight/generated_data/"

# Set seed
np.random.seed(42)

# Set number of samples
num_samples = [2100, 7000, 7000, 7000]

# Set source functions
def yh(cat,x):
    """High-fidelity wing weight response function."""
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = [x[:, i] for i in range(10)]
    Lam *= np.pi/180
    fac1 = (Wfw**0.0035)*((A/(np.cos(Lam)**2))**0.6)
    fac2 = (q**0.006)*(lam**0.04)*(((100*tc)/(np.cos(Lam)))**(-0.3))
    fac3 = (Nz*Wdg)**0.49
    return 0.036*(Sw**0.758)*fac1*fac2*fac3 + Sw*Wp
def yl1(cat,x):
    """Low-fidelity wing weight response function (variant 1)."""
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = [x[:, i] for i in range(10)]
    Lam *= np.pi/180
    fac1 = (Wfw**0.0035)*((A/(np.cos(Lam)**2))**0.6)
    fac2 = (q**0.006)*(lam**0.04)*(((100*tc)/(np.cos(Lam)))**(-0.3))
    fac3 = (Nz*Wdg)**0.49
    return 0.036*(Sw**0.758)*fac1*fac2*fac3 + 1*Wp
def yl2(cat,x):
    """Low-fidelity wing weight response function (variant 2)."""
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = [x[:, i] for i in range(10)]
    Lam *= np.pi/180
    fac1 = (Wfw**0.0035)*((A/(np.cos(Lam)**2))**0.6)
    fac2 = (q**0.006)*(lam**0.04)*(((100*tc)/(np.cos(Lam)))**(-0.3))
    fac3 = (Nz*Wdg)**0.49
    return 0.036*(Sw**0.8)*fac1*fac2*fac3 + 1*Wp
def yl3(cat,x):
    """Low-fidelity wing weight response function (variant 3)."""
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = [x[:, i] for i in range(10)]
    Lam *= np.pi/180
    fac1 = (Wfw**0.0035)*((A/(np.cos(Lam)**2))**0.6)
    fac2 = (q**0.006)*(lam**0.04)*(((100*tc)/(np.cos(Lam)))**(-0.3))
    fac3 = (Nz*Wdg)**0.49
    return 0.036*(Sw**0.9)*fac1*fac2*fac3 + 0*Wp
source_functions = [yh, yl1, yl2, yl3]

# Set numerical input ranges
num_ranges = [
    (150, 200),
    (220, 300),
    (6, 10),
    (-10, 10),
    (16, 45),
    (0.5, 1),
    (0.08, 0.18),
    (2.5, 6),
    (1700, 2500),
    (0.025, 0.08),
    ]

# Make dataset
dataset = data.Generate_Analytic_Dataset(
    dsource = 4,
    dcat = None,
    dnum = 10,
    dtargets = 1,
    qual_in = False,
    quant_in = True,
    num_samples = num_samples,
    source_functions = source_functions,
    num_ranges = num_ranges,
    noise_variance = [(25,), (25,), (25,), (25,)],
    random_generator = np.random.default_rng(42),
)

# Save dataset
dataset.save(save_path, "wing_weight_dataset")

# Load dataset
dataset = data.MultiFidelityDataset.load(save_path, "wing_weight_dataset")

# Split dataset
split_ratios = [15/2100, 6/2100, 0.99]
train_dataset, val_dataset, test_dataset = data.split_dataset(dataset, split_ratios)

# Save datasets
data.save_splits(train_dataset, val_dataset, test_dataset, save_path, "wing_weight_dataset")