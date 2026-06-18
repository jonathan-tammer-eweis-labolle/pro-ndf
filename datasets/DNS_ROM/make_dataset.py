"""
Generate and save the DNS_ROM dataset splits.
"""

import sys
import os
import numpy as np

# Add src directory to path to import package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from prondf import data
import dill

# Set save path (relative to project root)
save_path = "datasets/DNS_ROM/generated_data/"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset from pickle file
pickle_path = os.path.join(script_dir, 'DNS_ROM.pkl')
with open(pickle_path, 'rb') as f:
    dataset = dill.load(f)

# Get metadata
dsource = dataset['dsource'][0]
dnum = dataset['dnum']
dtargets = dataset['dy']

# Combine train, val, and test data
# Source (one-hot encoded)
source_train = dataset['t_train_OH']
source_val = dataset['t_val_OH']
source_test = dataset['t_test_OH']
source_all = np.vstack([source_train, source_val, source_test])

# Numerical inputs
num_train = dataset['x_train']
num_val = dataset['x_val']
num_test = dataset['x_test']
num_all = np.vstack([num_train, num_val, num_test])

# Targets
targets_train = dataset['y_train']
targets_val = dataset['y_val']
targets_test = dataset['y_test']
targets_all = np.vstack([targets_train, targets_val, targets_test])

# Count samples per source in the combined dataset
num_samples = []
for source_idx in range(dsource):
    source_mask = source_all[:, source_idx] == 1
    num_samples.append(np.sum(source_mask))

# Create metadata
meta = {
    'dsource': dsource,
    'dcat': None,
    'dnum': dnum,
    'dtargets': dtargets,
    'qual_in': False,
    'quant_in': True,
    'num_samples': num_samples,
}

# Create MultiFidelityDataset
dataset = data.MultiFidelityDataset(
    source=source_all,
    cat=None,
    num=num_all,
    targets=targets_all,
    meta=meta
)

# Save full dataset
dataset.save(save_path, "DNS_ROM_dataset")

# Load dataset to verify
dataset = data.MultiFidelityDataset.load(save_path, "DNS_ROM_dataset")

# Recreate the exact original splits by manually assigning indices
# Since we combined train+val+test in that order, the indices map directly:
# train: 0 to n_train_orig-1
# val: n_train_orig to n_train_orig+n_val_orig-1  
# test: n_train_orig+n_val_orig to end

n_train_orig = len(source_train)
n_val_orig = len(source_val)

train_indices_combined = np.arange(n_train_orig)
val_indices_combined = np.arange(n_train_orig, n_train_orig + n_val_orig)
test_indices_combined = np.arange(n_train_orig + n_val_orig, len(source_all))

# Create split datasets using the original indices
train_source = source_all[train_indices_combined, :]
train_num = num_all[train_indices_combined, :]
train_targets = targets_all[train_indices_combined, :]

val_source = source_all[val_indices_combined, :]
val_num = num_all[val_indices_combined, :]
val_targets = targets_all[val_indices_combined, :]

test_source = source_all[test_indices_combined, :]
test_num = num_all[test_indices_combined, :]
test_targets = targets_all[test_indices_combined, :]

# Count samples per source in each split for metadata
train_num_samples = []
val_num_samples = []
test_num_samples = []

for source_idx in range(dsource):
    train_mask = train_source[:, source_idx] == 1
    val_mask = val_source[:, source_idx] == 1
    test_mask = test_source[:, source_idx] == 1
    
    train_num_samples.append(np.sum(train_mask))
    val_num_samples.append(np.sum(val_mask))
    test_num_samples.append(np.sum(test_mask))

# Create metadata for each split
train_meta = meta.copy()
train_meta['num_samples'] = train_num_samples

val_meta = meta.copy()
val_meta['num_samples'] = val_num_samples

test_meta = meta.copy()
test_meta['num_samples'] = test_num_samples

# Create split datasets
train_dataset = data.MultiFidelityDataset(
    source=train_source,
    cat=None,
    num=train_num,
    targets=train_targets,
    meta=train_meta
)

val_dataset = data.MultiFidelityDataset(
    source=val_source,
    cat=None,
    num=val_num,
    targets=val_targets,
    meta=val_meta
)

test_dataset = data.MultiFidelityDataset(
    source=test_source,
    cat=None,
    num=test_num,
    targets=test_targets,
    meta=test_meta
)

# Calculate and print detailed split information
total_all = len(train_dataset) + len(val_dataset) + len(test_dataset)
train_pct = (len(train_dataset) / total_all) * 100
val_pct = (len(val_dataset) / total_all) * 100
test_pct = (len(test_dataset) / total_all) * 100

print(f"\nRecreated original splits:")
print(f"  Train: {len(train_dataset)} samples ({train_pct:.2f}%)")
print(f"  Val:   {len(val_dataset)} samples ({val_pct:.2f}%)")
print(f"  Test:  {len(test_dataset)} samples ({test_pct:.2f}%)")
print(f"  Total: {total_all} samples")

print(f"\nPer-source breakdown:")
for source_idx in range(dsource):
    n_train = train_num_samples[source_idx]
    n_val = val_num_samples[source_idx]
    n_test = test_num_samples[source_idx]
    n_total = n_train + n_val + n_test
    
    train_pct_src = (n_train / n_total) * 100 if n_total > 0 else 0
    val_pct_src = (n_val / n_total) * 100 if n_total > 0 else 0
    test_pct_src = (n_test / n_total) * 100 if n_total > 0 else 0
    
    print(f"  Source {source_idx}: Train={n_train} ({train_pct_src:.2f}%), Val={n_val} ({val_pct_src:.2f}%), Test={n_test} ({test_pct_src:.2f}%), Total={n_total}")

# Save splits
data.save_splits(train_dataset, val_dataset, test_dataset, save_path, "DNS_ROM_dataset")