# Poisson 1D Neural Network with Conformal Prediction - Streamlit App

This Streamlit application visualizes uncertainty quantification for 1D Poisson equation using different conformal prediction methods. The application uses **py-pde** to generate realistic solutions to the Poisson equation.

## Features

- **Realistic PDE Data Generation**: Uses py-pde to solve the 1D Poisson equation with configurable forcing terms and boundary conditions
- **Multiple Methods**: Compare three conformal prediction approaches:
  - Conformalised Quantile Regression (CQR)
  - Absolute Error Residual-based Conformal Prediction (AER)
  - Standard Deviation/Dropout-based Conformal Prediction (STD)
- **Interactive Visualization**:
  - View prediction bounds as lines (not error bars)
  - Adjust alpha parameter in real-time
  - Compare empirical coverage across methods
  - Analyze calibration curves
- **Flexible Configuration**:
  - Adjustable grid size (16-128 points)
  - Configurable forcing term bounds
  - Variable training epochs and data splits

## Physics Background

The app solves the 1D Poisson equation:

```
-d²u/dx² = f(x)
```

with boundary conditions:
- u(0) = 0 (Dirichlet)
- du/dx(1) = 1 (Neumann)

where `f(x)` is a constant forcing term sampled uniformly from a specified range.

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit numpy torch matplotlib py-pde scipy
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_poisson_cp.py
```

Or if you prefer to specify the port:
```bash
streamlit run streamlit_poisson_cp.py --server.port 8501
```

## How to Use the App

1. **Configure Settings** (Left Sidebar):
   
   **Data Settings:**
   - Set number of samples for training/calibration/testing
   
   **PDE Settings:**
   - Forcing Lower Bound: Minimum value for constant forcing term (default: 0.0)
   - Forcing Upper Bound: Maximum value for constant forcing term (default: 4.0)
   - Grid Size: Number of spatial points (16-128, default: 32)
   
   **Training Settings:**
   - Choose number of training epochs (50-500)
   
   **Conformal Prediction:**
   - Select alpha value (miscoverage rate)
   - Choose which methods to compare

2. **Generate Data**:
   - Click "Generate Data" button
   - py-pde will solve the Poisson equation for each sample
   - Progress bar shows generation status
   - Data will be split into train/cal/test sets

3. **Train Models**:
   - Click "Train" button for each method you want to use
   - Training progress is shown with a progress bar
   - You can train one or all three methods

4. **Visualize Results** (Right Panel):
   - **Prediction Bounds Tab**: View uncertainty bounds as lines for each method with color-coded fills
   - **Coverage Analysis Tab**: Compare empirical vs target coverage with bar charts
   - **Alpha Sensitivity Tab**: See how coverage changes across different alpha values (calibration curve)

## Key Parameters

- **Alpha (α)**: Miscoverage rate. Coverage = 1 - α
  - α = 0.1 means 90% coverage
  - α = 0.05 means 95% coverage

- **Forcing Bounds**: Range of constant forcing terms
  - Larger range = more diverse solutions
  - Default [0, 4] provides good variety

- **Grid Size**: Spatial resolution
  - More points = higher resolution (but slower training)
  - Default 32 is a good balance

- **Training Samples**: More samples = better model performance
- **Calibration Samples**: More samples = tighter bounds
- **Epochs**: More epochs = better convergence (diminishing returns after ~200)

## Methods Explained

1. **CQR (Conformalised Quantile Regression)** - Red Theme 🔴:
   - Trains two models (5th and 95th percentile quantiles)
   - Provides adaptive bounds based on quantile predictions
   - Good for asymmetric uncertainties

2. **AER (Absolute Error Residual)** - Teal Theme 🔵:
   - Trains a single regression model
   - Uses absolute residuals for calibration
   - Symmetric bounds around prediction
   - Computationally efficient

3. **STD (Standard Deviation/Dropout)** - Purple Theme 🟣:
   - Uses Monte Carlo Dropout for uncertainty estimation
   - Estimates epistemic uncertainty
   - Bounds based on prediction variance
   - Good for capturing model uncertainty

## Color Scheme

The app uses a professional color palette with distinct colors for each method:
- **Ground Truth**: Dark slate gray
- **CQR**: Ruby red gradient
- **AER**: Teal/cyan gradient
- **STD**: Purple/amethyst gradient

## Tips

- Start with default parameters to understand the app
- Train all three methods to compare their performance
- Lower alpha values give wider bounds (higher coverage)
- The calibration curve should be close to the diagonal for well-calibrated methods
- Try different forcing bounds to see how the methods handle different data distributions
- Increase grid size for smoother solutions (at the cost of computation time)

## Technical Notes

- The py-pde library provides accurate finite-difference solutions to the Poisson equation
- Neural networks are trained on CPU by default
- Models use 3 hidden layers with 64 neurons each
- Dropout rate is set to 0.1 for the STD method
- All three methods use the same calibration procedure for fair comparison