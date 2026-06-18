# Wing Weight Dataset

## Overview
This dataset is based on the **Wing Weight function**, a widely-used benchmark function in computer experiments and engineering design optimization. The function models the weight of a light aircraft wing as a function of design parameters and is commonly used for testing metamodeling and multi-fidelity methods.

## Source Function
The high-fidelity wing weight function calculates the weight (lbs) of an aircraft wing:

```
yh = 0.036·Sw^0.758·fac1·fac2·fac3 + Sw·Wp

where:
fac1 = Wfw^0.0035 · (A/cos²(Λ))^0.6
fac2 = q^0.006 · λ^0.04 · ((100·tc)/cos(Λ))^(-0.3)
fac3 = (Nz·Wdg)^0.49
```

Note: Λ (sweep angle) is converted from degrees to radians in the calculation.

## Input Variables (10 dimensions)
The wing weight function has 10 continuous input variables:

| Variable | Description | Range | Units |
|----------|-------------|-------|-------|
| Sw | Wing area | [150, 200] | ft² |
| Wfw | Weight of fuel in wing | [220, 300] | lb |
| A | Aspect ratio | [6, 10] | - |
| Λ (Lam) | Quarter-chord sweep angle | [-10, 10] | degrees |
| q | Dynamic pressure at cruise | [16, 45] | lb/ft² |
| λ (lam) | Taper ratio | [0.5, 1] | - |
| tc | Aerofoil thickness to chord ratio | [0.08, 0.18] | - |
| Nz | Ultimate load factor | [2.5, 6] | - |
| Wdg | Flight design gross weight | [1700, 2500] | lb |
| Wp | Paint weight | [0.025, 0.08] | lb/ft² |

## Multi-Source Configuration

### Number of Sources: 4
This dataset includes one high-fidelity source and three low-fidelity approximations:

1. **yh (High-Fidelity)**: Standard wing weight function with Sw·Wp term
2. **yl1 (Low-Fidelity 1)**: Simplified with 1·Wp instead of Sw·Wp, RRMSE of 0.20 relative to high-fidelity source
3. **yl2 (Low-Fidelity 2)**: Modified with Sw^0.8 instead of Sw^0.758, and 1·Wp, RRMSE of 1.14 relative to high-fidelity source
4. **yl3 (Low-Fidelity 3)**: Modified with Sw^0.9 instead of Sw^0.758, and 0·Wp (no paint weight term), RRMSE of 5.75 relative to high-fidelity source

Where RRMSE (relative root mean squared error) is measured on 10,000 random test samples and indicates accuracy of the low-fidelity source in modeling the high-fidelity source.

## Dataset Statistics

### Total Dataset
- **Total samples**: 23,100 (2,100 + 7,000 + 7,000 + 7,000)
- **Samples per source**: [2100, 7000, 7000, 7000]
- **Noise variance**: 25.0 for all sources
- **Random seed**: 42

### Data Splits
- **Training set**: 165 samples (15 + 50 + 50 + 50)
  - Split ratio: 15/2100 from high-fidelity
- **Validation set**: 86 samples (6 samples from high-fidelity source)
  - Split ratio: 6/2100 from high-fidelity
- **Test set**: Remaining ~99% (approximately 22,849 samples)
  - Split ratio: 0.99 of remaining data

### Input/Output Configuration
- **Qualitative inputs**: None
- **Quantitative inputs**: 10 continuous variables
- **Output dimensions**: 1 (wing weight in lbs)

## Purpose
The wing weight function is frequently used to test:
- Multi-fidelity modeling approaches
- Surrogate modeling methods
- Engineering design optimization
- Global sensitivity analysis
- Uncertainty quantification

The multiple source approximations allow for testing hierarchical or multi-fidelity learning algorithms where cheaper low-fidelity evaluations can be used to improve predictions of expensive high-fidelity evaluations.

## References
- Mora, C., Eweis-LaBolle, J. T., Johnson, T., Gadde, L., & Bostanabad, R. (2023). Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets. Computer Methods in Applied Mechanics and Engineering, 415, 116207. https://doi.org/10.1016/j.cma.2023.116207
- This function is commonly found in benchmark suites for engineering design optimization and surrogate modeling.
