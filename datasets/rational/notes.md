# Rational Dataset

## Overview
This dataset is a simple 1-dimensional synthetic benchmark formulated as a basic test for multi-fidelity modeling methods. It consists of rational functions (ratios of polynomials) with varying levels of polynomial complexity to simulate different fidelity levels.

## Source Functions
The dataset contains four source functions based on rational functions with polynomial denominators:

### High-Fidelity Function
```
yh(x) = 1 / (0.1·x³ + x² + x + 1)
```

### Low-Fidelity Functions
```
yl1(x) = 1 / (0.2·x³ + x² + x + 1)    [Modified cubic coefficient]
yl2(x) = 1 / (x² + x + 1)              [No cubic term]
yl3(x) = 1 / (x² + 1)                  [Only quadratic and constant]
```

## Input Variables (1 dimension)
This is a single-input dataset:

| Variable | Description | Range |
|----------|-------------|-------|
| x | Continuous scalar input | [-2.0, 3.0] |

## Multi-Source Configuration

### Number of Sources: 4
This dataset includes one high-fidelity source and three low-fidelity approximations:

1. **yh (High-Fidelity)**: Full cubic rational function with coefficient 0.1 for x³
2. **yl1 (Low-Fidelity 1)**: Modified with coefficient 0.2 for x³ instead of 0.1, RRMSE of 0.23 relative to high-fidelity source
3. **yl2 (Low-Fidelity 2)**: Simplified by removing cubic term entirely, RRMSE of 0.15 relative to high-fidelity source
4. **yl3 (Low-Fidelity 3)**: Further simplified by removing both cubic and linear terms, RRMSE of 0.73 relative to high-fidelity source

Where RRMSE (relative root mean squared error) is measured on 10,000 random test samples and indicates accuracy of the low-fidelity source in modeling the high-fidelity source.

The progression of low-fidelity approximations demonstrates systematic simplification of the polynomial structure, making this useful for testing how multi-fidelity methods handle varying degrees of model discrepancy.

## Dataset Statistics

### Total Dataset
- **Total samples**: 15,200 (800 + 4,800 + 4,800 + 4,800)
- **Samples per source**: [800, 4800, 4800, 4800]
- **Noise variance**: 0.001 for all sources
- **Random seed**: 42

### Data Splits
- **Training set**: 95 samples (5 + 30 + 30 + 30)
  - Split ratio: 5/800 from high-fidelity
- **Validation set**: 3 samples (from high-fidelity source)
  - Split ratio: 3/800 from high-fidelity
- **Test set**: Remaining ~99% (approximately 15,102 samples)
  - Split ratio: 0.99 of remaining data

### Input/Output Configuration
- **Qualitative inputs**: None
- **Quantitative inputs**: 1 continuous variable
- **Output dimensions**: 1 (rational function value)

## Purpose
This simple 1-dimensional dataset serves as a basic test case for:
- Multi-fidelity modeling approaches
- Neural density field methods
- Hierarchical surrogate modeling
- Algorithm debugging and validation
- Quick prototyping and proof-of-concept testing

The low dimensionality and controlled structure make it ideal for:
- Visualizing model predictions and uncertainties
- Understanding algorithm behavior in a simple setting
- Rapid iteration during method development
- Sanity checking implementations before scaling to higher-dimensional problems

## Characteristics
- **Simplicity**: Single input dimension allows for easy visualization and interpretation
- **Controlled complexity**: Systematic removal of polynomial terms creates interpretable fidelity hierarchy
- **Smooth functions**: All source functions are smooth and continuous over the input range
- **Low noise**: Small noise variance (0.001) allows focus on model bias rather than noise handling
- **Balanced data**: More low-fidelity samples than high-fidelity, mimicking practical multi-fidelity scenarios

This dataset provides a foundational test bed for multi-fidelity methods before applying them to more complex, higher-dimensional problems like the borehole or wing weight functions.

## References
- Mora, C., Eweis-LaBolle, J. T., Johnson, T., Gadde, L., & Bostanabad, R. (2023). Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets. Computer Methods in Applied Mechanics and Engineering, 415, 116207. https://doi.org/10.1016/j.cma.2023.116207
