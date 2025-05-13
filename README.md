# in-context-denoising

[![arXiv](https://img.shields.io/badge/arXiv-2502.05164-b31b1b.svg)](https://arxiv.org/abs/2502.05164)

Code for the paper "In-context denoising with one-layer transformers: connections between attention and associative memory retrieval" (ICML 2025).

This repository implements a framework connecting attention-based architectures with dense associative memory (DAM) networks. We demonstrate that certain denoising problems can be solved optimally with a single-layer transformer, where the trained attention layer performs a single gradient descent update on a context-aware DAM energy landscape.

#### Dependencies
- Python >=3.9 with libraries listed in `requirements.txt`
- Install dependencies: `pip install -r requirements.txt`

#### Entry-point and basic usage
- Train a model on a task with `src/nn_train_methods.py` (modify main and run) 
- Visualize low-dim data for each case (linear/spheres/GMM) using `scripts/overview_datagen.ipynb` 

#### Figure generation (general steps to reproduce)
- **Figure 3**
  - run `src/nn_train_ensemble.py` to train models with different seeds (do this for each case)   
  - run `scripts/replot_multitraj_loss.py`, pointing to pairs of output directories from step 1

- **Figure 4a**
  - run `src/nn_train_ensemble.py` to train models at varying context length (linear subspace task)
  - run `scripts/replot_multitraj_vary_contextlen.py`, pointing to the output directory from 1

- **Figure 4b**
  - run `scripts/analysis_case_linear_inference_dim_d.ipynb`, train a new model or point to trained model

- **Figure 5**
  - run `scripts/analysis_energy_landscape_traj.py` with appropriate settings

#### `src/` : core scripts
- `settings.py`:          Global settings and defaults for the project
- `nn_model_base.py`:     Base classes for the different networks
- `nn_train_methods.py`:  Integrates the datagen + training loop
- `nn_train_ensemble.py`: Training script for loss spread across multiple runs; also supports varying context length
- `nn_loss_baselines.py`: Baseline loss functions
- `data_io.py`:           Data I/O utilities
- `data_tools.py`:        Data processing utilities
- `torch_device.py`:      Sets torch device
- `visualize.py`:         Visualization utilities

#### `scripts/`: Analysis / plotting scripts and notebooks
- `analysis_energy_landscape.py`: Analysis script for energy landscape
- `analysis_case_linear_inference_dim_d.ipynb`: Load a network trained on the linear self-attention task and analyze the inference performance with varying subspace dimension d
- `replot_multitraj_loss.py`:            Replot results of `nn_train_ensemble.py` (for ensemble of diff seeds)
- `replot_multitraj_vary_contextlen.py`: Replot results of `nn_train_ensemble.py` (for a varying context length ensemble)
- `replot_weights_timeseries.py`:  Replot a timeseries of weights (each epoch of training) given a run directory
- `overview_datagen.ipynb`: Visualize low-dim data for each case (linear/spheres/GMM)
