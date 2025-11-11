# Uncertainty Quantification of Surrogate Models using Conformal Prediction

This repository contains the implementation of experiments from the paper **["Uncertainty Quantification of Surrogate Models using Conformal Prediction"]([url](https://arxiv.org/abs/2408.09881))**. The code provides a framework for obtaining statistically guaranteed error bars on neural network predictions for spatio-temporal PDE systems using conformal prediction (CP).

## Overview

Conformal prediction provides **distribution-free, model-agnostic uncertainty quantification** with guaranteed coverage. This repository demonstrates CP across four classical PDEs, showing how to:

- Train surrogate models (MLPs, U-Nets, FNOs) for PDE systems
- Apply three nonconformity scoring methods (CQR, AER, STD)
- Obtain calibrated prediction intervals with guaranteed coverage
- Validate coverage on independent test sets
- Handle out-of-distribution scenarios

**Key Result**: CP provides valid uncertainty bounds regardless of model architecture or training procedure, requiring only exchangeability between calibration and test data.

## Experiment Structure

All experiments follow this 6-step framework:

```
1. Generate/gather training data (simulation or experimental)
2. Train surrogate model (or use pre-trained model)
3. Generate/gather calibration dataset
4. Compute nonconformity scores via CP framework
5. Construct prediction sets with guaranteed coverage (1-α)
6. Validate coverage on independent test set
```

## Repository Structure

```
.
├── Poisson/                    # 1D Poisson equation experiments
├── Conv_Diff/                  # 1D Convection-Diffusion experiments
├── Wave/                       # 2D Wave equation experiments
├── Navier_Stokes/             # 2D Navier-Stokes experiments (FNO data)
└── README.md
```

## Experiments

### 1. 1D Poisson Equation
**Physics**: Steady-state elliptic PDE modeling electrostatics, gravitation, and potential fields.

**Domain**: 1D spatial domain [0, 1] discretized into 32 points

**Data Generation**:
- 7,000 simulations using finite difference methods
- Training: 5,000 samples
- Calibration: 1,000 samples
- Validation: 1,000 samples
- Parameter: Initial field value u_init ~ U(0, 4)

**Model**: 3-layer MLP (64 neurons per layer)

**Training**:
```bash
cd Poisson/
python Poisson_MLP_CP.py
```

**Results**: All methods achieve ~90% coverage with tight error bars (AER: 0.002 normalized units)

---

### 2. 1D Convection-Diffusion Equation
**Physics**: Spatio-temporal system combining diffusion (smoothing) and convection (transport).

**Domain**: 
- Spatial: x ∈ [0, 10] with 200 points
- Temporal: t ∈ [0, 0.1] with 100 time steps (downsampled to 20)

**Data Generation**:
- 5,000 simulations using forward-time centered-space finite difference
- Training: 3,000 samples (Latin hypercube sampling)
- Calibration: 1,000 samples (out-of-distribution)
- Validation: 1,000 samples (out-of-distribution)

**Sampling Parameters**:

*Training*:
| Parameter | Domain | Type |
|-----------|--------|------|
| Diffusion Coefficient (D) | [sin(x/π), sin(x/2π)] | Continuous |
| Convection velocity (c) | [0.1, 0.5] | Continuous |
| Mean (μ) | [1.0, 8.0] | Continuous |
| Variance (σ²) | [0.25, 0.75] | Continuous |

*Calibration/Validation (Out-of-Distribution)*:
| Parameter | Domain | Type |
|-----------|--------|------|
| Diffusion Coefficient (D) | [sin(x/2π), sin(x/4π)] | Continuous |
| Convection velocity (c) | [0.5, 1.0] | Continuous |
| Mean (μ) | [1.0, 8.0] | Continuous |
| Variance (σ²) | [0.25, 0.75] | Continuous |

**Model**: 1D U-Net (4 encoder-decoder levels, batch normalization, tanh activation)
- Maps first 10 time steps → next 10 time steps

**Training**:
```bash
cd Conv_Diff/
python CD_UNet_CP.py
```

**Results**: CP successfully calibrates even with distribution shift:
- CQR: 25.53% → 93.05% coverage after calibration
- STD: 88.43% → 90.29% coverage after calibration

---

### 3. 2D Wave Equation
**Physics**: 2D wave equation modeling wave propagation in acoustics, optics, and quantum mechanics.

**Domain**:
- Spatial: [−1, 1]² with 33×33 grid
- Temporal: t ∈ [0, 1] with 80 time steps
- Wave velocity: c (constant for training, c/2 for out-of-distribution testing)

**Data Generation**:
- Spectral solver with leapfrog time discretization
- Chebyshev spectral method for spatial discretization
- Training: 500 simulations
- Calibration: 1,000 simulations
- Validation: 1,000 simulations

**Initial Condition Parameters** (Latin hypercube sampling):
| Parameter | Domain | Type |
|-----------|--------|------|
| Amplitude (α) | [10, 50] | Continuous |
| X position (β) | [0.1, 0.5] | Continuous |
| Y position (γ) | [0.1, 0.5] | Continuous |

**Models**:

1. **U-Net** (Feed-forward):
   - Maps 20 input time steps → 30 output time steps
   - Output shape: [30, 33, 33]
   - Architecture: 2 encoder-decoder levels with batch normalization

2. **FNO** (Autoregressive):
   - Maps 20 initial time steps → recursively predicts next 10 steps
   - Total output: 60 time steps
   - Output shape: [60, 33, 33]
   - 6 Fourier layers with width 32

**Training**:
```bash
cd Wave/
# U-Net
python Wave_Unet_CP.py

# FNO
python Wave_FNO_CP.py

# Out-of-distribution testing (half speed)
python tests/Wave_Unet_CP_halfspeed.py
python tests/Wave_FNO_CP_halfspeed.py
```

**Results**: 
- U-Net: ~90% coverage with all methods (AER most efficient)
- FNO: Tighter coverage than U-Net, maintains validity under distribution shift

---

### 4. 2D Navier-Stokes Equations
**Physics**: Incompressible 2D Navier-Stokes equations modeling viscous fluid dynamics (vorticity formulation).

**Domain**:
- Spatial: (0, 1) × (0, 1) grid
- Temporal evolution of vorticity field

**Data Source**: Uses data from [FNO paper](https://github.com/neuraloperator/neuraloperator) (Li et al., 2021)

**Setup**:
- Training: viscosity ν = 10⁻³
- Calibration/Testing: viscosity ν = 10⁻⁴ (out-of-distribution)
- Maps 10 input time steps → next 10 output time steps

**Model**: Fourier Neural Operator (FNO)
- Modified with dropout layers for STD-based CP
- 6 Fourier layers
- Modes: 12 (spectral truncation)
- Width: configurable

**Training**:
```bash
cd Navier_Stokes/
python NS_FNO_CP.py

# Training details in utils.py and model_FNO.py
```

**Results**: 
- Uncalibrated dropout: only 7.52% coverage
- After CP calibration: 90.27% coverage (STD method)
- Demonstrates CP's ability to correct severely underestimated uncertainty

---

## Conformal Prediction Methods

The repository implements three nonconformity scoring methods:

### 1. Conformalised Quantile Regression (CQR)
**Description**: Train three separate models to predict quantiles (5th, 50th, 95th percentiles).

**Nonconformity Score**: `s(x,y) = max{q̂_low(x) - y, y - q̂_high(x)}`

**Prediction Set**: `[q̂_low(x) - q̂, q̂_high(x) + q̂]`

**Training Loss**: Quantile loss (pinball loss)

**Requirements**: 
- 3 models (or 1 model with 3 outputs)
- No architectural modifications needed

### 2. Absolute Error Residual (AER)
**Description**: Train a single deterministic model, compute absolute residuals.

**Nonconformity Score**: `s(x,y) = |y - ŷ(x)|`

**Prediction Set**: `[ŷ(x) - q̂, ŷ(x) + q̂]`

**Training Loss**: MSE or MAE

**Requirements**: 
- 1 model
- No architectural modifications
- **Most computationally efficient**

### 3. Standard Deviation (STD)
**Description**: Use probabilistic models that output mean and standard deviation.

**Nonconformity Score**: `s(x,y) = |y - μ(x)| / σ(x)`

**Prediction Set**: `[μ(x) - q̂σ(x), μ(x) + q̂σ(x)]`

**Training Loss**: MSE (with dropout) or negative log-likelihood

**Requirements**: 
- 1 probabilistic model
- Architectural modifications: dropout layers (rate 0.1-0.2)
- Monte Carlo sampling at inference (typically 100 samples)

---

## Training Your Own Models

### General Training Configuration

All experiments use similar training setups:

```python
configuration = {
    "Epochs": 500,
    "Batch Size": 50,
    "Optimizer": 'Adam',
    "Learning Rate": 0.005,
    "Scheduler Step": 100,
    "Scheduler Gamma": 0.5,
    "Normalisation Strategy": 'Min-Max',  # Scale to [-1, 1]
}
```

### Training Procedure

1. **For CQR**: Train three models with quantile loss
   ```python
   # Train models for 5th, 50th, 95th percentiles
   loss = quantile_loss(predictions, targets, quantile=[0.05, 0.5, 0.95])
   ```

2. **For AER**: Train one model with MSE loss
   ```python
   loss = F.mse_loss(predictions, targets)
   ```

3. **For STD**: Train one model with dropout + MSE loss
   ```python
   model = UNetWithDropout(dropout_rate=0.1)
   loss = F.mse_loss(predictions, targets)
   ```

### Data Normalization

All experiments use Min-Max normalization to scale field values to [-1, 1]:

```python
def normalize(data):
    return 2 * (data - data.min()) / (data.max() - data.min()) - 1

def denormalize(normalized_data, original_min, original_max):
    return (normalized_data + 1) / 2 * (original_max - original_min) + original_min
```

---

## Performing Conformal Prediction

### Step-by-Step Guide

#### Step 1: Load Trained Model and Data

```python
import torch
import numpy as np
from model import UNet1d  # or your model architecture

# Load model
model = UNet1d(T_in=10, step=10, width=32)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Load data
cal_inputs = ...   # Shape: (n_cal, T_in, spatial_dims...)
cal_targets = ...  # Shape: (n_cal, T_out, spatial_dims...)
test_inputs = ...  # Shape: (n_test, T_in, spatial_dims...)
test_targets = ... # Shape: (n_test, T_out, spatial_dims...)
```

#### Step 2: Define Nonconformity Score Function

**For AER:**
```python
def compute_nonconformity_scores(predictions, targets):
    """
    Compute absolute error residuals
    
    Args:
        predictions: model predictions (n_samples, T_out, ...)
        targets: ground truth (n_samples, T_out, ...)
    
    Returns:
        scores: nonconformity scores (n_samples, T_out, ...)
    """
    return np.abs(targets - predictions)
```

**For CQR:**
```python
def compute_nonconformity_scores_cqr(pred_lower, pred_upper, targets):
    """
    Compute CQR nonconformity scores
    
    Args:
        pred_lower: lower quantile predictions (n_samples, T_out, ...)
        pred_upper: upper quantile predictions (n_samples, T_out, ...)
        targets: ground truth (n_samples, T_out, ...)
    
    Returns:
        scores: nonconformity scores (n_samples, T_out, ...)
    """
    return np.maximum(pred_lower - targets, targets - pred_upper)
```

**For STD:**
```python
def compute_nonconformity_scores_std(mean, std, targets, n_samples=100):
    """
    Compute STD-normalized nonconformity scores
    
    Args:
        mean: predicted mean (n_samples, T_out, ...)
        std: predicted standard deviation (n_samples, T_out, ...)
        targets: ground truth (n_samples, T_out, ...)
    
    Returns:
        scores: normalized nonconformity scores (n_samples, T_out, ...)
    """
    return np.abs(targets - mean) / (std + 1e-10)
```

#### Step 3: Compute Calibration Quantile

```python
def calibrate(cal_scores, n_cal, alpha):
    """
    Compute the (1-alpha) quantile of calibration scores
    
    Args:
        cal_scores: nonconformity scores from calibration set
        n_cal: number of calibration samples
        alpha: desired miscoverage rate (e.g., 0.1 for 90% coverage)
    
    Returns:
        q_hat: calibrated quantile threshold
    """
    # Compute quantile level (accounts for finite sample correction)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    
    # Compute quantile across calibration set (cell-wise for spatio-temporal data)
    q_hat = np.quantile(cal_scores, q_level, axis=0, method='higher')
    
    return q_hat
```

#### Step 4: Generate Predictions on Calibration Set

```python
# Generate calibration predictions
with torch.no_grad():
    cal_predictions = model(torch.FloatTensor(cal_inputs)).numpy()

# Compute calibration scores
cal_scores = compute_nonconformity_scores(cal_predictions, cal_targets)

# Calibrate
alpha = 0.1  # 90% coverage
n_cal = cal_inputs.shape[0]
q_hat = calibrate(cal_scores, n_cal, alpha)
```

#### Step 5: Construct Prediction Sets on Test Data

**For AER:**
```python
# Generate test predictions
with torch.no_grad():
    test_predictions = model(torch.FloatTensor(test_inputs)).numpy()

# Construct prediction intervals
prediction_sets = [
    test_predictions - q_hat,  # Lower bound
    test_predictions + q_hat   # Upper bound
]
```

**For STD (with Monte Carlo Dropout):**
```python
def predict_with_dropout(model, inputs, n_samples=100):
    """Generate predictions with MC dropout"""
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(torch.FloatTensor(inputs)).numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std

# Generate test predictions with uncertainty
test_mean, test_std = predict_with_dropout(model, test_inputs, n_samples=100)

# Construct prediction intervals
prediction_sets = [
    test_mean - q_hat * test_std,  # Lower bound
    test_mean + q_hat * test_std   # Upper bound
]
```

#### Step 6: Validate Coverage

```python
def empirical_coverage(prediction_sets, targets):
    """
    Compute empirical coverage
    
    Args:
        prediction_sets: [lower_bounds, upper_bounds]
        targets: ground truth
    
    Returns:
        coverage: fraction of points within prediction intervals
    """
    lower, upper = prediction_sets
    in_interval = (targets >= lower) & (targets <= upper)
    coverage = in_interval.mean()
    
    return coverage

# Validate
coverage = empirical_coverage(prediction_sets, test_targets)
print(f"Empirical coverage: {coverage:.4f}")
print(f"Target coverage: {1-alpha:.4f}")
```

#### Complete Example Script

```python
import torch
import numpy as np
from model import UNet1d

# Configuration
alpha = 0.1  # 90% coverage
T_in, T_out = 10, 10
width = 32

# Load model
model = UNet1d(T_in, T_out, width)
model.load_state_dict(torch.load('trained_model.pth'))

# Load data
cal_inputs = np.load('cal_inputs.npy')
cal_targets = np.load('cal_targets.npy')
test_inputs = np.load('test_inputs.npy')
test_targets = np.load('test_targets.npy')

# Step 1: Calibration predictions
model.eval()
with torch.no_grad():
    cal_preds = model(torch.FloatTensor(cal_inputs)).numpy()

# Step 2: Compute nonconformity scores
cal_scores = np.abs(cal_targets - cal_preds)

# Step 3: Calibrate
n_cal = cal_inputs.shape[0]
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
q_hat = np.quantile(cal_scores, q_level, axis=0, method='higher')

# Step 4: Test predictions
with torch.no_grad():
    test_preds = model(torch.FloatTensor(test_inputs)).numpy()

# Step 5: Prediction sets
lower = test_preds - q_hat
upper = test_preds + q_hat

# Step 6: Validate coverage
in_interval = (test_targets >= lower) & (test_targets <= upper)
coverage = in_interval.mean()

print(f"Target coverage: {1-alpha:.2%}")
print(f"Empirical coverage: {coverage:.2%}")
print(f"Average interval width: {(upper - lower).mean():.4f}")
```

---

## Key Implementation Details

### Exchangeability Requirement

CP requires **exchangeability** between calibration and test data. This means:

✅ **Valid**:
- Calibration and test from same distribution
- Calibration and test from different distributions (out-of-distribution CP)
- Using pre-trained models with new calibration data

❌ **Invalid**:
- Temporal data where test comes after calibration in time series
- Calibration data from biased/selected subset

### Cell-wise Calibration

For spatio-temporal outputs, CP is applied **independently** to each cell:

```python
# predictions shape: (n_samples, T_out, Nx, Ny)
# Each cell (t, x, y) gets its own calibrated quantile

cal_scores = np.abs(targets - predictions)  # Shape: (n_cal, T_out, Nx, Ny)
q_hat = np.quantile(cal_scores, q_level, axis=0)  # Shape: (T_out, Nx, Ny)
```

This provides **marginal coverage** at every spatio-temporal location.

### Finite Sample Correction

The quantile level includes finite sample correction:

```python
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
```

For large n_cal, this approaches (1 - alpha). For small n_cal, it provides conservative bounds.

### Out-of-Distribution CP

CP can provide valid coverage even when test distribution differs from training:

1. Train model on distribution P_train
2. Calibrate on distribution P_cal (can differ from P_train)
3. Test on distribution P_test

**Requirement**: P_cal and P_test must be exchangeable (not P_train and P_test)

Example: Convection-Diffusion experiment
- Training: D ∈ [sin(x/π), sin(x/2π)], c ∈ [0.1, 0.5]
- Cal/Test: D ∈ [sin(x/2π), sin(x/4π)], c ∈ [0.5, 1.0]

---

## Computational Requirements

### Training
- GPU recommended (experiments used NVIDIA A100)
- Training time varies by problem:
  - Poisson: ~5 minutes
  - Convection-Diffusion: ~30 minutes
  - Wave (U-Net): ~1 hour
  - Wave (FNO): ~2 hours
  - Navier-Stokes: ~4 hours

### Calibration
- CPU sufficient (experiments used standard laptop)
- Calibration time per method:
  - AER: <10 seconds (fastest)
  - CQR: <30 seconds
  - STD: <500 seconds (slowest due to MC sampling)

### Memory
- Depends on dataset size and spatial resolution
- Typical requirements:
  - 1D problems: 4-8 GB RAM
  - 2D problems: 16-32 GB RAM

---

## Dependencies

```bash
# Core dependencies
torch>=1.12.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0

# PDE simulation
py-pde>=0.23.0  # For Poisson equation

# Optional (for specific experiments)
h5py>=3.0.0  # For loading FNO datasets
```

Install all dependencies:
```bash
pip install torch numpy scipy matplotlib py-pde h5py
```

---

## Reproducing Paper Results

To reproduce the results in Table 1 of the paper:

```bash
# 1D Poisson
cd Poisson && python Poisson_MLP_CP.py

# 1D Convection-Diffusion  
cd Conv_Diff && python CD_UNet_CP.py

# 2D Wave (U-Net)
cd Wave && python Wave_Unet_CP.py

# 2D Wave (FNO)
cd Wave && python Wave_FNO_CP.py

# 2D Navier-Stokes
cd Navier_Stokes && python NS_FNO_CP.py
```

Each script will:
1. Load or generate data
2. Train model(s) for each CP method
3. Perform calibration
4. Compute coverage and tightness metrics
5. Generate visualization plots

---

## Key Results

| Experiment | Model | Method | Uncal. Coverage | Calib. Coverage | Cal. Time (s) | Tightness |
|------------|-------|--------|----------------|----------------|---------------|-----------|
| Poisson | MLP | AER | - | 90.05% | 0.003 | 0.002 |
| Conv-Diff* | U-Net | CQR | 25.53% | 93.05% | 19.70 | 0.314 |
| Conv-Diff* | U-Net | AER | - | 92.60% | 8.30 | 0.266 |
| Wave | U-Net | AER | - | 94.91% | 3.52 | 0.013 |
| Wave* | FNO | AER | - | 89.24% | 34.18 | 0.330 |
| Navier-Stokes* | FNO | STD | 7.52% | 90.27% | 64.75 | 0.448 |

*Out-of-distribution evaluation

**Key Takeaways**:
1. All methods achieve ~90% target coverage after CP calibration
2. AER is fastest and works well when architectures are flexible
3. CP successfully handles distribution shift (marked with *)
4. Uncalibrated methods can severely underestimate uncertainty (e.g., 7.52% → 90.27%)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gopakumar2025uncertainty,
  title={Uncertainty Quantification of Surrogate Models using Conformal Prediction},
  author={Gopakumar, Vignesh and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Additional Resources

- **Conformal Prediction**: 
  - [Tutorial by Angelopoulos & Bates](https://people.eecs.berkeley.edu/~angelopoulos/blog/posts/gentle-intro/)
  - [Original paper by Vovk et al.](https://link.springer.com/book/10.1007/978-3-031-06649-8)

- **Neural Operators**:
  - [FNO paper (Li et al., 2021)](https://arxiv.org/abs/2010.08895)
  - [Neural Operator repository](https://github.com/neuraloperator/neuraloperator)

---

## License

[Specify your license here]

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your contact information].
