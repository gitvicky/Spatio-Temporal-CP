# Poisson 1D Neural Network with Conformal Prediction - Streamlit App

This Streamlit application visualizes uncertainty quantification for 1D Poisson equation using different conformal prediction methods.

## Features

- **Data Generation**: Generate synthetic Poisson 1D equation data
- **Multiple Methods**: Compare three conformal prediction approaches:
  - Conformalised Quantile Regression (CQR)
  - Residual-based Conformal Prediction
  - Dropout-based Conformal Prediction
- **Interactive Visualization**:
  - View prediction bounds as lines (not error bars)
  - Adjust alpha parameter in real-time
  - Compare empirical coverage across methods
  - Analyze calibration curves

## Installation

```bash
# Install required packages
pip install streamlit numpy torch matplotlib --break-system-packages
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
   - Set number of samples for training/calibration/testing
   - Choose number of training epochs
   - Select alpha value (miscoverage rate)
   - Choose which methods to compare

2. **Generate Data**:
   - Click "Generate Data" button
   - Data will be split into train/cal/test sets

3. **Train Models**:
   - Click "Train" button for each method you want to use
   - Training progress is shown with a progress bar
   - You can train one or all three methods

4. **Visualize Results** (Right Panel):
   - **Prediction Bounds Tab**: View uncertainty bounds as lines for each method
   - **Coverage Analysis Tab**: Compare empirical vs target coverage
   - **Alpha Sensitivity Tab**: See how coverage changes across different alpha values (calibration curve)

## Key Parameters

- **Alpha (α)**: Miscoverage rate. Coverage = 1 - α
  - α = 0.1 means 90% coverage
  - α = 0.05 means 95% coverage

- **Training Samples**: More samples = better model performance
- **Calibration Samples**: More samples = tighter bounds
- **Epochs**: More epochs = better convergence (diminishing returns after ~200)

## Methods Explained

1. **CQR (Conformalised Quantile Regression)**:
   - Trains two models (lower and upper quantiles)
   - Provides adaptive bounds based on quantile predictions

2. **Residual**:
   - Trains a single regression model
   - Uses absolute residuals for calibration
   - Symmetric bounds around prediction

3. **Dropout**:
   - Uses Monte Carlo Dropout for uncertainty estimation
   - Estimates epistemic uncertainty
   - Bounds based on prediction variance

## Tips

- Start with default parameters to understand the app
- Train all three methods to compare their performance
- Lower alpha values give wider bounds (higher coverage)
- The calibration curve should be close to the diagonal for well-calibrated methods
