# HOIP Dataset

## Overview
This dataset is based on the **HOIP (Hybrid Organic-Inorganic Perovskite) problem**, which focuses on predicting the inter-molecular binding energy in hybrid organic-inorganic perovskite crystals. The problem combines multiple sources of data with different fidelities to model the binding energy of these materials.

## Problem Description
The goal is to predict inter-molecular binding energy by combining four sources of data:
1. **High-fidelity (HF)**: High-fidelity data source
2. **Low-fidelity 1 (LF1)**: Low-fidelity data source
3. **Low-fidelity 2 (LF2)**: Low-fidelity data source
4. **Low-fidelity 3 (LF3)**: Low-fidelity data source

The low-fidelity sources provide computationally cheaper approximations with unknown levels of fidelity relative to the high-fidelity source.

## Input Variables

### Categorical Inputs (3 dimensions)
The dataset has 3 categorical input variables that correspond to the elements present in each crystal:

| Variable | Description | Number of Levels |
|----------|-------------|------------------|
| Categorical input 1 | Element type 1 | 10 levels (l1 = 10) |
| Categorical input 2 | Element type 2 | 3 levels (l2 = 3) |
| Categorical input 3 | Element type 3 | 16 levels (l3 = 16) |

### Numerical Inputs
- **Numerical inputs**: None

## Multi-Source Configuration

### Number of Sources: 4
This dataset includes one high-fidelity source and three low-fidelity approximations:

1. **Source 0 (High-Fidelity)**: High-fidelity data source
2. **Source 1 (Low-Fidelity 1)**: Low-fidelity data source 1
3. **Source 2 (Low-Fidelity 2)**: Low-fidelity data source 2
4. **Source 3 (Low-Fidelity 3)**: Low-fidelity data source 3

The low-fidelity sources provide computationally cheaper approximations that trade off accuracy for speed, with unknown levels of fidelity relative to the high-fidelity source.

## Dataset Statistics

### Total Dataset
- **Total samples**: 1379
- **Samples per source**: [480, 480, 179, 240]
- **Number of sources**: 4
- **Categorical input dimensions**: 3 (with 10, 3, and 16 levels respectively)
- **Numerical input dimensions**: None (no numerical inputs)
- **Output dimensions**: 1 (inter-molecular binding energy)

### Data Splits
The dataset preserves the exact split ratios from the original source dataset. The splits are recreated exactly as provided in the original pickle file. Run `make_dataset.py` to see the exact split counts printed to the console.

- **Training set**: 1103 samples (80% of total)
  - Source 0: 384 samples
  - Source 1: 384 samples
  - Source 2: 143 samples
  - Source 3: 192 samples
- **Validation set**: 138 samples (10% of total)
  - Source 0: 48 samples
  - Source 1: 48 samples
  - Source 2: 18 samples
  - Source 3: 24 samples
- **Test set**: 138 samples (10% of total)
  - Source 0: 48 samples
  - Source 1: 48 samples
  - Source 2: 18 samples
  - Source 3: 24 samples

## Purpose
The HOIP dataset is used to test:
- Multi-fidelity modeling approaches
- Data fusion methods for combining high and low-fidelity data
- Categorical input handling in multi-fidelity models
- Computational efficiency vs. accuracy trade-offs in material property prediction

The multiple source approximations allow for testing hierarchical or multi-fidelity learning algorithms where cheaper low-fidelity evaluations can be used to improve predictions of expensive high-fidelity evaluations while reducing computational costs.

## References
- Mora, C., Eweis-LaBolle, J. T., Johnson, T., Gadde, L., & Bostanabad, R. (2023). Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets. Computer Methods in Applied Mechanics and Engineering, 415, 116207. https://doi.org/10.1016/j.cma.2023.116207