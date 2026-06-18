# ProNDF
> Probabilistic Neural Data Fusion

This repo is a refactor and extension of Pro-NDF, a method I helped develop during my Ph.D.



Probabilistic Neural Data Fusion for multi-fidelity modeling with a focus on
interpretable manifold learning and principled uncertainty quantification.

This project is a refactor of the original Pro-NDF introduced in
“Probabilistic neural data fusion for learning from an arbitrary number of
multi-fidelity data sets” and extends the methodology with a parameterized
distribution output and additional loss-weighting functionality, including
the Two-Moment Weighting scheme from my dissertation work.

Paper: https://arxiv.org/pdf/2301.13271

## Why this project exists
Multi-fidelity data is common in science and engineering: you often have a small
amount of high-fidelity (expensive) data and a larger amount of low-fidelity
(cheap) data. ProNDF reframes multi-fidelity modeling as a manifold learning
problem, enabling joint learning from an arbitrary number of sources while
quantifying source-wise discrepancies and predictive uncertainty.

## Methodology overview
ProNDF is a three-block architecture:
- **Block 1 (source manifold, probabilistic)** maps a one-hot source indicator
  to a low-dimensional latent space, making inter-source similarity interpretable.
- **Block 2 (categorical manifold, deterministic)** maps categorical inputs to a
  latent space when qualitative features are present.
- **Block 3 (predictive model, probabilistic output)** maps the latent variables
  and numerical inputs to a parameterized output distribution.

Training uses a composite loss with likelihood-based terms and optional
regularizers. The refactor emphasizes parameterized output distributions (for
aleatoric uncertainty) and supports flexible loss weighting, including the
Two-Moment Weighting algorithm for balancing loss components.

## What’s different from the original ProNDF
- **Parameterized output distributions** instead of a full BNN architecture.
  The model outputs distribution parameters directly (e.g., Gaussian mean and
  variance) for robust uncertainty quantification.
- **Two-Moment Weighting** for multi-objective loss balancing, designed for
  stable training when data are scarce or unbalanced.
- **Cleaner modularization** with registries for blocks, losses, optimizers, and
  loggers, enabling experimentation without touching core training logic.

## Repository structure
- `src/prondf/models.py`: `ProNDF` LightningModule and `Build_ProNDF` constructor.
- `src/prondf/blocks.py`: deterministic and probabilistic MLP blocks.
- `src/prondf/losses.py`: loss functions, data splits, loss handlers, and weighting algorithms.
- `src/prondf/data.py`: dataset abstraction, dataset I/O, splitting, and preprocessing utilities.
- `src/prondf/plotting.py`: helper plots for evaluation and diagnostics.
- `src/prondf/loggers.py`: modular training/plot loggers (TensorBoard-compatible).
- `datasets/`: example datasets and notes (see dataset-specific README/notes files).
- `notebooks/examples.ipynb` / `notebooks/tests.ipynb`: primary usage and experiment notebooks.

## Usage (notebook-first)
Most usage happens through notebooks (see `examples.ipynb` and `tests.ipynb`).
The code is also structured for scripted/CLI usage via `Build_ProNDF`.

Minimal example (conceptual):
```python
from prondf.models import Build_ProNDF
from prondf.data import load_splits

train_ds, val_ds, test_ds = load_splits("datasets/wing_weight", "wing_weight_dataset")
model = Build_ProNDF(dataset_meta=train_ds.meta)
```

## Datasets
This repo includes several benchmark datasets (analytic and real-world). Each
dataset folder contains notes with details on provenance and setup. The README
focuses on methodology, so dataset specifics are kept minimal here.

## Logging (optional)
Custom loggers are implemented in `loggers.py` and are compatible with
PyTorch Lightning’s TensorBoard logger. If you use the loggers, you should
instantiate a Lightning logger (e.g., `TensorBoardLogger`) when training.

## Installation (minimal)
This repo uses a standard `src/` layout and can be installed in editable mode:
`pip install -e .`

Expected dependencies include:
- `torch`, `pytorch_lightning`
- `numpy`, `scipy`, `matplotlib`

If you want a formal install process, I can add a `pyproject.toml` or
`requirements.txt`.

## License
No license is currently specified. Standard practice for research portfolios is
to choose a permissive license (e.g., MIT or BSD-3-Clause). If you want, I can
add one and include a `LICENSE` file.

## Citation
If you use this code or the original Pro-NDF methodology, please cite:

```bibtex
@article{mora2023prondf,
  title={Probabilistic Neural Data Fusion for Learning from an Arbitrary Number of Multi-fidelity Data Sets},
  author={Mora, Carlos and Eweis-Labolle, Jonathan Tammer and Johnson, Tyler and Gadde, Likith and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2301.13271},
  year={2023}
}
```

## Acknowledgements
This refactor is based on the Pro-NDF architecture in the original paper and
extends it for improved robustness and flexibility for research workflows.
