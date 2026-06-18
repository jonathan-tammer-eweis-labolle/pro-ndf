# DNS-ROM Dataset

## Overview
This dataset is based on the **DNS-ROM (Direct Numerical Simulation - Reduced Order Model) problem**, which focuses on predicting the toughness of a multiscale metallic component with spatially varying porosity. The problem combines multiple sources of data with different fidelities to balance accuracy against computational costs.

## Problem Description
The goal is to predict material toughness by combining four sources of data:
1. **High-fidelity**: Direct numerical simulations (DNS)
2. **Low-fidelity 1**: Reduced-order model (ROM) with 800 clusters
3. **Low-fidelity 2**: Reduced-order model (ROM) with 1600 clusters
4. **Low-fidelity 3**: Reduced-order model (ROM) with 3200 clusters

The more clusters are used in the ROM, the more similar are its results compared to those of DNS at the expense of a higher computational burden.

## Input Variables (6 dimensions)
The dataset has 6 continuous numerical input variables:

| Variable | Description |
|----------|-------------|
| Pore volume fraction | Volume fraction of pores in the material |
| Number of pores | Total count of pores |
| Pore aspect ratio | Ratio describing pore shape |
| Average nearest neighbor distance | Average distance between neighboring pores |
| Evolutionary rate parameter | Parameter governing damage response under load |
| Critical effective plastic strain | Parameter governing damage response under load |

The last two inputs (evolutionary rate parameter and critical effective plastic strain) govern the damage response of the material under load.

## Multi-Source Configuration

### Number of Sources: 4
This dataset includes one high-fidelity source and three low-fidelity approximations:

1. **Source 0 (High-Fidelity)**: Direct numerical simulations (DNS) - 70 samples
2. **Source 1 (Low-Fidelity 1)**: ROM with 800 clusters - 110 samples
3. **Source 2 (Low-Fidelity 2)**: ROM with 1600 clusters - 170 samples
4. **Source 3 (Low-Fidelity 3)**: ROM with 3200 clusters - 250 samples

The low-fidelity ROM sources provide computationally cheaper approximations that trade off accuracy for speed. As the number of clusters increases, the ROM results become more similar to the high-fidelity DNS results.

## Dataset Statistics

### Total Dataset
- **Total samples**: 600 (70 + 110 + 170 + 250)
- **Samples per source**: [70, 110, 170, 250]
- **Number of sources**: 4
- **Numerical input dimensions**: 6
- **Output dimensions**: 1 (material toughness)

### Data Splits
The dataset preserves the exact split ratios from the original source dataset. The splits are recreated exactly as provided in the original pickle file. Run `make_dataset.py` to see the exact split counts printed to the console.

- **Training set**: 431 samples (71.83% of total)
  - Source 0: 50 samples
  - Source 1: 79 samples
  - Source 2: 122 samples
  - Source 3: 180 samples
- **Validation set**: 49 samples (8.17% of total)
  - Source 0: 6 samples
  - Source 1: 9 samples
  - Source 2: 14 samples
  - Source 3: 20 samples
- **Test set**: 120 samples (20% of total)
  - Source 0: 14 samples
  - Source 1: 22 samples
  - Source 2: 34 samples
  - Source 3: 50 samples

### Input/Output Configuration
- **Qualitative inputs**: None
- **Quantitative inputs**: 6 continuous variables
- **Output dimensions**: 1 (material toughness)

## Purpose
The DNS-ROM dataset is used to test:
- Multi-fidelity modeling approaches
- Reduced-order modeling techniques
- Data fusion methods for combining high and low-fidelity simulations
- Computational efficiency vs. accuracy trade-offs in material modeling

The multiple source approximations allow for testing hierarchical or multi-fidelity learning algorithms where cheaper low-fidelity evaluations can be used to improve predictions of expensive high-fidelity evaluations while reducing computational costs.

## References
- Deng, S., Mora, C., Apelian, D., & Bostanabad, R. (2022). Data-Driven Calibration of Multi-Fidelity Multiscale Fracture Models. *Journal of Engineering Materials and Technology*, 144(4), 041003. https://doi.org/10.1115/1.4055951
- Mora, C., Eweis-LaBolle, J. T., Johnson, T., Gadde, L., & Bostanabad, R. (2023). Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets. Computer Methods in Applied Mechanics and Engineering, 415, 116207. https://doi.org/10.1016/j.cma.2023.116207
