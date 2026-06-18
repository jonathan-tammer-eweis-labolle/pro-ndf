# Borehole Dataset

## Overview
This dataset is based on the **Borehole function**, a widely-used benchmark function in computer experiments and uncertainty quantification. The function models water flow through a borehole and is commonly used for testing metamodeling and multi-fidelity methods.

## Source Function
The high-fidelity borehole function calculates water flow rate (m³/yr) through a borehole:

```
        2π·Tu·(Hu - Hl)
yh = ─────────────────────────────────────────
     log(r/rw)·(1 + (2·L·Tu)/(log(r/rw)·rw²·Kw) + Tu/Tl)
```

## Input Variables (8 dimensions)
The borehole function has 8 continuous input variables:

| Variable | Description | Range | Units |
|----------|-------------|-------|-------|
| Tu | Transmissivity of upper aquifer | [100, 1000] | m²/yr |
| Hu | Potentiometric head of upper aquifer | [990, 1110] | m |
| Hl | Potentiometric head of lower aquifer | [700, 820] | m |
| r | Radius of influence | [100, 10000] | m |
| rw | Radius of borehole | [0.05, 0.15] | m |
| Tl | Transmissivity of lower aquifer | [10, 500] | m²/yr |
| L | Length of borehole | [1000, 2000] | m |
| Kw | Hydraulic conductivity of borehole | [6000, 12000] | m/yr |

## Multi-Source Configuration

### Number of Sources: 5
This dataset includes one high-fidelity source and four low-fidelity approximations:

1. **yh (High-Fidelity)**: Standard borehole function
2. **yl1 (Low-Fidelity 1)**: Modified with Hl → 0.8·Hl, L → 1·L, RRMSE of 3.67 relative to high-fidelity source
3. **yl2 (Low-Fidelity 2)**: Modified with Hl → 3·Hl, L → 8·L, Tl → 0.75·Tl, RRMSE of 3.73 relative to high-fidelity source
4. **yl3 (Low-Fidelity 3)**: Modified with Hu → 1.1·Hu, r → 4·r, L → 3·L, RRMSE of 0.38 relative to high-fidelity source
5. **yl4 (Low-Fidelity 4)**: Modified with Hu → 1.05·Hu, r → 2·r, L → 2·L, RRMSE of 0.19 relative to high-fidelity source

Where RRMSE (relative root mean squared error) is measured on 10,000 random test samples and indicates accuracy of the low-fidelity source in modeling the high-fidelity source.

## Dataset Statistics

### Total Dataset
- **Total samples**: 30,100 (2,100 + 7,000 + 7,000 + 7,000 + 7,000)
- **Samples per source**: [2100, 7000, 7000, 7000, 7000]
- **Noise variance**: 6.25 for all sources
- **Random seed**: 42

### Data Splits
- **Training set**: 215 samples (15 + 50 + 50 + 50 + 50)
  - Split ratio: 15/2100 from high-fidelity
- **Validation set**: 86 samples (6 samples from high-fidelity source)
  - Split ratio: 6/2100 from high-fidelity
- **Test set**: Remaining ~99% (approximately 29,799 samples)
  - Split ratio: 0.99 of remaining data

### Input/Output Configuration
- **Qualitative inputs**: None
- **Quantitative inputs**: 8 continuous variables
- **Output dimensions**: 1 (water flow rate)

## Purpose
The borehole function is frequently used to test:
- Multi-fidelity modeling approaches
- Surrogate modeling methods
- Uncertainty quantification techniques
- Global sensitivity analysis

The multiple source approximations allow for testing hierarchical or multi-fidelity learning algorithms where cheaper low-fidelity evaluations can be used to improve predictions of expensive high-fidelity evaluations.

## References
- Mora, C., Eweis-LaBolle, J. T., Johnson, T., Gadde, L., & Bostanabad, R. (2023). Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets. Computer Methods in Applied Mechanics and Engineering, 415, 116207. https://doi.org/10.1016/j.cma.2023.116207
- This function is commonly found in benchmark suites for computer experiments and surrogate modeling.
