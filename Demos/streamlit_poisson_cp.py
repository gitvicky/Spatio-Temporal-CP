#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App for Poisson 1D Neural Network with Conformal Prediction
Visualize uncertainty bounds for different methods as alpha varies
Enhanced with professional color schemes
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from timeit import default_timer
import os

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# ===========================
# Model Definitions
# ===========================

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLP_Dropout(nn.Module):
    """MLP with Dropout for uncertainty estimation"""
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout_rate=0.1):
        super(MLP_Dropout, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def quantile_loss(pred, target, gamma=0.5):
    """Quantile loss function"""
    error = target - pred
    return torch.max((gamma - 1) * error, gamma * error)


def MLP_dropout_eval(model, x, num_samples=50):
    """Evaluate MLP with dropout for uncertainty estimation"""
    model.train()  # Keep dropout active
    predictions = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred


# ===========================
# Color Palette Configuration
# ===========================

# Primary Color Scheme (current)
COLOR_PALETTE = {
    'CQR': {
        'main': '#C41E3A',        # Ruby Red
        'prediction': '#E63946',   # Bright Red
        'bounds': '#F07178',       # Light Coral
        'fill': '#FFD3D8',         # Pale Pink
        'dark': '#8B1538'          # Dark Ruby
    },
    'AER': {
        'main': '#006D77',         # Dark Cyan
        'prediction': '#06A77D',   # Teal
        'bounds': '#4ECDC4',       # Light Turquoise
        'fill': '#A8E6CF',         # Mint
        'dark': '#004D57'          # Deep Teal
    },
    'STD': {
        'main': '#5B2C6F',         # Deep Purple
        'prediction': '#8E44AD',   # Amethyst
        'bounds': '#A569BD',       # Medium Purple
        'fill': '#D7BDE2',         # Lavender
        'dark': '#4A235A'          # Dark Purple
    },
    'ground_truth': '#2C3E50',     # Dark Slate
    'background': '#F8F9FA',        # Light Gray
    'grid': '#E0E0E0'              # Medium Gray
}

# Alternative Color Scheme (uncomment to use)
# COLOR_PALETTE = {
#     'CQR': {
#         'main': '#D32F2F',        # Red 700
#         'prediction': '#EF5350',   # Red 400
#         'bounds': '#E57373',       # Red 300
#         'fill': '#FFCDD2',         # Red 100
#         'dark': '#B71C1C'          # Red 900
#     },
#     'AER': {
#         'main': '#F57C00',         # Orange 700
#         'prediction': '#FF9800',   # Orange 500
#         'bounds': '#FFB74D',       # Orange 300
#         'fill': '#FFE0B2',         # Orange 100
#         'dark': '#E65100'          # Orange 900
#     },
#     'STD': {
#         'main': '#1976D2',         # Blue 700
#         'prediction': '#2196F3',   # Blue 500
#         'bounds': '#64B5F6',       # Blue 300
#         'fill': '#BBDEFB',         # Blue 100
#         'dark': '#0D47A1'          # Blue 900
#     },
#     'ground_truth': '#263238',     # Blue Grey 900
#     'background': '#FAFAFA',       # Grey 50
#     'grid': '#E0E0E0'              # Grey 300
# }


# ===========================
# Data Generation using py-pde
# ===========================

def generate_poisson_data(n_samples=4000, lb=0, ub=4, grid_size=32):
    """Generate Poisson equation data using py-pde
    
    Solves the 1D Poisson equation: -d²u/dx² = f(x)
    with boundary conditions: u(0) = 0, du/dx(1) = 1
    
    Args:
        n_samples: Number of samples to generate
        lb: Lower bound for uniform forcing term
        ub: Upper bound for uniform forcing term
        grid_size: Number of grid points (default: 32)
    
    Returns:
        X: Input forcing terms (n_samples, grid_size)
        Y: Solution fields (n_samples, grid_size)
    """
    st.info(f"Generating {n_samples} Poisson 1D samples using py-pde...")
    
    from pde import CartesianGrid, ScalarField, solve_poisson_equation
    
    # Generate random forcing term parameters
    np.random.seed(42)
    params = lb + (ub - lb) * np.random.uniform(size=n_samples)
    
    inps = []
    outs = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for ii in range(n_samples):
        # Create grid and forcing field
        grid = CartesianGrid([[0, 1]], grid_size, periodic=False)
        field = ScalarField(grid, params[ii])
        
        # Solve Poisson equation with boundary conditions
        # BC: u(0) = 0 (Dirichlet), du/dx(1) = 1 (Neumann)
        result = solve_poisson_equation(field, bc=[{"value": 0}, {"derivative": 1}])
        
        inps.append(field.data)
        outs.append(result.data)
        
        # Update progress
        if (ii + 1) % 100 == 0 or ii == n_samples - 1:
            progress = (ii + 1) / n_samples
            progress_bar.progress(progress)
            progress_text.text(f"Generated {ii + 1}/{n_samples} samples")
    
    progress_text.empty()
    
    X = np.asarray(inps, dtype=np.float32)
    Y = np.asarray(outs, dtype=np.float32)
    
    st.success(f"✅ Generated {n_samples} samples using py-pde")
    
    return X, Y


# ===========================
# Training Functions
# ===========================

def train_quantile_model(X_train, Y_train, gamma, epochs=100, batch_size=100):
    """Train quantile regression model"""
    device = torch.device('cpu')
    
    # Get dimensions from data
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    model = MLP(input_dim, output_dim, 3, 64)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(Y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            out = model(xx)
            loss = quantile_loss(out, yy, gamma=gamma).pow(2).mean()
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_history.append(epoch_loss / len(train_loader))
        progress_bar.progress((epoch + 1) / epochs)
    
    return model, loss_history


def train_residual_model(X_train, Y_train, epochs=100, batch_size=100):
    """Train residual-based model"""
    device = torch.device('cpu')
    
    # Get dimensions from data
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    model = MLP(input_dim, output_dim, 3, 64)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    loss_func = torch.nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(Y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            out = model(xx)
            loss = loss_func(out, yy)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_history.append(epoch_loss / len(train_loader))
        progress_bar.progress((epoch + 1) / epochs)
    
    return model, loss_history


def train_dropout_model(X_train, Y_train, epochs=100, batch_size=100):
    """Train dropout-based model"""
    device = torch.device('cpu')
    
    # Get dimensions from data
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    model = MLP_Dropout(input_dim, output_dim, 3, 64, dropout_rate=0.1)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    loss_func = torch.nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(Y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            
            out = model(xx)
            loss = loss_func(out, yy)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        loss_history.append(epoch_loss / len(train_loader))
        progress_bar.progress((epoch + 1) / epochs)
    
    return model, loss_history


# ===========================
# Conformal Prediction
# ===========================

def compute_cqr_scores(model_lower, model_upper, X_cal, Y_cal):
    """Compute CQR calibration scores"""
    with torch.no_grad():
        lower_pred = model_lower(torch.FloatTensor(X_cal)).numpy()
        upper_pred = model_upper(torch.FloatTensor(X_cal)).numpy()
    
    cal_scores = np.maximum(Y_cal - upper_pred, lower_pred - Y_cal)
    return cal_scores


def compute_residual_scores(model, X_cal, Y_cal):
    """Compute residual-based calibration scores"""
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_cal)).numpy()
    
    cal_scores = np.abs(Y_cal - pred)
    return cal_scores


def compute_dropout_scores(model, X_cal, Y_cal):
    """Compute dropout-based calibration scores"""
    mean_cal, std_cal = MLP_dropout_eval(model, torch.FloatTensor(X_cal))
    
    cal_upper = mean_cal + std_cal
    cal_lower = mean_cal - std_cal
    
    cal_scores = np.maximum(Y_cal - cal_upper, cal_lower - Y_cal)
    return cal_scores


def get_prediction_bounds(model_type, models, X_test, cal_scores, alpha):
    """Get prediction bounds for a given alpha"""
    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, axis=0, method='higher')
    
    if model_type == "CQR":
        model_lower, model_upper = models
        with torch.no_grad():
            lower = model_lower(torch.FloatTensor(X_test)).numpy()
            upper = model_upper(torch.FloatTensor(X_test)).numpy()
        return lower - qhat, upper + qhat
    
    elif model_type == "AER":
        model = models[0]
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test)).numpy()
        return pred - qhat, pred + qhat
    
    elif model_type == "STD":
        model = models[0]
        mean_pred, std_pred = MLP_dropout_eval(model, torch.FloatTensor(X_test))
        lower = mean_pred - std_pred
        upper = mean_pred + std_pred
        return lower - qhat, upper + qhat


def compute_coverage(Y_true, lower_bound, upper_bound):
    """Compute empirical coverage"""
    coverage = ((Y_true >= lower_bound) & (Y_true <= upper_bound)).mean()
    return coverage


# ===========================
# Streamlit App
# ===========================

def main():
    st.set_page_config(page_title="Poisson 1D CP Visualizer", layout="wide")
    
    st.title("🔬 Poisson 1D Neural Network with Conformal Prediction")
    st.markdown("Visualize uncertainty bounds for different conformal prediction methods")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Data settings
    st.sidebar.subheader("Data Settings")
    n_samples = st.sidebar.number_input("Total Samples", value=4000, min_value=1000, step=1000)
    train_split = st.sidebar.number_input("Training Samples", value=2000, min_value=1000, step=500)
    cal_split = st.sidebar.number_input("Calibration Samples", value=1000, min_value=100, step=100)
    
    # Py-PDE settings
    st.sidebar.subheader("PDE Settings")
    lb = st.sidebar.number_input("Forcing Lower Bound", value=0.0, step=0.5)
    ub = st.sidebar.number_input("Forcing Upper Bound", value=8.0, step=0.5)
    grid_size = st.sidebar.number_input("Grid Size", value=32, min_value=16, max_value=128, step=16)
    
    # Training settings
    st.sidebar.subheader("Training Settings")
    epochs = st.sidebar.slider("Epochs", min_value=50, max_value=500, value=100, step=50)
    
    # Alpha parameter
    st.sidebar.subheader("Conformal Prediction")
    alpha = st.sidebar.slider("Alpha (1-α = coverage)", min_value=0.10, max_value=0.90, value=0.50, step=0.10)
    
    # Method selection
    methods = st.sidebar.multiselect(
        "Select Methods",
        ["CQR", "AER", "STD"],
        default=["CQR", "AER", "STD"]
    )
    
    # Visualization sample
    viz_idx = st.sidebar.number_input("Visualization Sample Index", value=50, min_value=0, step=1)
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = {}
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📊 Data & Training")
        
        # Generate data button
        if st.button("🔄 Generate Data", type="primary"):
            with st.spinner("Generating data..."):
                X, Y = generate_poisson_data(n_samples, lb=lb, ub=ub, grid_size=grid_size)
                
                # Split data
                X_train = X[:train_split]
                Y_train = Y[:train_split]
                X_cal = X[train_split:train_split + cal_split]
                Y_cal = Y[train_split:train_split + cal_split]
                X_test = X[train_split + cal_split:]
                Y_test = Y[train_split + cal_split:]
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.Y_train = Y_train
                st.session_state.X_cal = X_cal
                st.session_state.Y_cal = Y_cal
                st.session_state.X_test = X_test
                st.session_state.Y_test = Y_test
                st.session_state.data_generated = True
                
                st.success(f"✅ Data generated: {len(X_train)} train, {len(X_cal)} cal, {len(X_test)} test")
        
        # Train models
        if st.session_state.data_generated:
            st.markdown("---")
            st.subheader("🎯 Train Models")
            
            for method in methods:
                if st.button(f"Train {method}", key=f"train_{method}"):
                    with st.spinner(f"Training {method} model..."):
                        X_train = st.session_state.X_train
                        Y_train = st.session_state.Y_train
                        X_cal = st.session_state.X_cal
                        Y_cal = st.session_state.Y_cal
                        
                        if method == "CQR":
                            st.write("Training lower quantile (γ=0.05)...")
                            model_lower, loss_lower = train_quantile_model(X_train, Y_train, gamma=0.05, epochs=epochs)
                            st.write("Training upper quantile (γ=0.95)...")
                            model_upper, loss_upper = train_quantile_model(X_train, Y_train, gamma=0.95, epochs=epochs)
                            
                            # Compute calibration scores
                            cal_scores = compute_cqr_scores(model_lower, model_upper, X_cal, Y_cal)
                            
                            st.session_state.models_trained[method] = {
                                'models': (model_lower, model_upper),
                                'cal_scores': cal_scores,
                                'loss': (loss_lower, loss_upper)
                            }
                            
                        elif method == "AER":
                            model, loss = train_residual_model(X_train, Y_train, epochs=epochs)
                            
                            # Compute calibration scores
                            cal_scores = compute_residual_scores(model, X_cal, Y_cal)
                            
                            st.session_state.models_trained[method] = {
                                'models': (model,),
                                'cal_scores': cal_scores,
                                'loss': (loss,)
                            }
                            
                        elif method == "STD":
                            model, loss = train_dropout_model(X_train, Y_train, epochs=epochs)
                            
                            # Compute calibration scores
                            cal_scores = compute_dropout_scores(model, X_cal, Y_cal)
                            
                            st.session_state.models_trained[method] = {
                                'models': (model,),
                                'cal_scores': cal_scores,
                                'loss': (loss,)
                            }
                        
                        st.success(f"✅ {method} model trained!")
            
            # Show training status
            if st.session_state.models_trained:
                st.markdown("---")
                st.subheader("📈 Training Status")
                for method in methods:
                    if method in st.session_state.models_trained:
                        st.success(f"✓ {method}")
                    else:
                        st.warning(f"✗ {method} (not trained)")
    
    with col2:
        st.header("📈 Visualization")
        
        if st.session_state.models_trained:
            tabs = st.tabs(["Prediction Bounds", "Coverage Analysis", "Alpha Sensitivity"])
            
            # Tab 1: Prediction Bounds
            with tabs[0]:
                st.subheader(f"Prediction Bounds (α={alpha:.2f}, 1-α={1-alpha:.2f})")
                
                X_test = st.session_state.X_test
                Y_test = st.session_state.Y_test
                
                # Ensure viz_idx is valid
                viz_idx = min(viz_idx, len(X_test) - 1)
                
                x_viz = X_test[viz_idx:viz_idx+1]
                y_viz = Y_test[viz_idx]
                grid_size = X_test.shape[1]  # Get grid size from data
                x_range = np.linspace(0, 1, grid_size)
                
                fig, axes = plt.subplots(len(st.session_state.models_trained), 1, 
                                        figsize=(12, 4.5*len(st.session_state.models_trained)))
                
                if len(st.session_state.models_trained) == 1:
                    axes = [axes]
                
                for idx, (method, data) in enumerate(st.session_state.models_trained.items()):
                    ax = axes[idx]
                    
                    # Get prediction bounds
                    lower, upper = get_prediction_bounds(
                        method, data['models'], x_viz, data['cal_scores'], alpha
                    )
                    
                    # Get mean prediction
                    if method == "CQR":
                        with torch.no_grad():
                            pred_lower = data['models'][0](torch.FloatTensor(x_viz)).numpy()
                            pred_upper = data['models'][1](torch.FloatTensor(x_viz)).numpy()
                        mean_pred = (pred_lower + pred_upper) / 2
                    elif method == "AER":
                        with torch.no_grad():
                            mean_pred = data['models'][0](torch.FloatTensor(x_viz)).numpy()
                    elif method == "STD":
                        mean_pred, _ = MLP_dropout_eval(data['models'][0], torch.FloatTensor(x_viz))
                    
                    # Get method-specific colors
                    colors = COLOR_PALETTE[method]
                    
                    # Plot with enhanced styling
                    ax.plot(x_range, y_viz, color=COLOR_PALETTE['ground_truth'], 
                           label='Ground Truth', linewidth=3, alpha=0.95, zorder=5)
                    ax.plot(x_range, mean_pred.flatten(), color=colors['prediction'], 
                           label='Prediction', linewidth=2.5, linestyle='--', alpha=0.9, zorder=4)
                    ax.plot(x_range, lower.flatten(), color=colors['bounds'], 
                           label='Lower Bound', linewidth=2, alpha=0.85, zorder=3)
                    ax.plot(x_range, upper.flatten(), color=colors['bounds'], 
                           label='Upper Bound', linewidth=2, alpha=0.85, zorder=3)
                    ax.fill_between(x_range, lower.flatten(), upper.flatten(), 
                                   alpha=0.3, color=colors['fill'], zorder=1)
                    
                    ax.set_xlabel('x', fontsize=12, fontweight='bold')
                    ax.set_ylabel('u(x)', fontsize=12, fontweight='bold')
                    ax.set_title(f'{method} Method (α={alpha:.2f})', 
                               fontsize=13, fontweight='bold', pad=12)
                    ax.legend(loc='best', framealpha=0.95, edgecolor='#CCCCCC', 
                            fontsize=10, shadow=True)
                    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, color=COLOR_PALETTE['grid'])
                    ax.set_facecolor(COLOR_PALETTE['background'])
                    
                    # Improve spine visibility
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#CCCCCC')
                        spine.set_linewidth(1.2)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 2: Coverage Analysis
            with tabs[1]:
                st.subheader("Empirical Coverage Analysis")
                
                X_test = st.session_state.X_test
                Y_test = st.session_state.Y_test
                
                coverage_data = []
                
                for method, data in st.session_state.models_trained.items():
                    lower, upper = get_prediction_bounds(
                        method, data['models'], X_test, data['cal_scores'], alpha
                    )
                    coverage = compute_coverage(Y_test, lower, upper)
                    coverage_data.append({
                        'Method': method,
                        'Empirical Coverage': f"{coverage:.4f}",
                        'Target Coverage': f"{1-alpha:.4f}",
                        'Difference': f"{coverage - (1-alpha):.4f}"
                    })
                
                st.table(coverage_data)
                
                # Plot coverage with enhanced colors
                fig, ax = plt.subplots(figsize=(11, 7))
                
                methods_list = [d['Method'] for d in coverage_data]
                emp_coverage = [float(d['Empirical Coverage']) for d in coverage_data]
                target = 1 - alpha
                
                x_pos = np.arange(len(methods_list))
                
                # Create bars with method-specific colors
                bars = []
                for i, method in enumerate(methods_list):
                    bar = ax.bar(i, emp_coverage[i], 
                               color=COLOR_PALETTE[method]['main'], 
                               alpha=0.8, edgecolor=COLOR_PALETTE[method]['dark'],
                               linewidth=2)
                    bars.append(bar)
                
                ax.axhline(y=target, color='#E63946', linestyle='--', 
                          linewidth=2.5, label=f'Target (1-α={target:.2f})', alpha=0.9)
                
                ax.set_xlabel('Method', fontsize=13, fontweight='bold')
                ax.set_ylabel('Coverage', fontsize=13, fontweight='bold')
                ax.set_title('Empirical Coverage vs Target Coverage', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(methods_list, fontsize=11, fontweight='bold')
                ax.legend(fontsize=11, framealpha=0.95, edgecolor='#CCCCCC', shadow=True)
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, 
                       color=COLOR_PALETTE['grid'], axis='y')
                ax.set_facecolor(COLOR_PALETTE['background'])
                
                # Improve spine visibility
                for spine in ax.spines.values():
                    spine.set_edgecolor('#CCCCCC')
                    spine.set_linewidth(1.2)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Tab 3: Alpha Sensitivity
            with tabs[2]:
                st.subheader("Coverage vs Alpha (Calibration Curve)")
                
                alpha_levels = np.arange(0.05, 0.95, 0.05)
                X_test = st.session_state.X_test
                Y_test = st.session_state.Y_test
                
                fig, ax = plt.subplots(figsize=(11, 9))
                
                # Plot ideal line
                ax.plot(1 - alpha_levels, 1 - alpha_levels, 
                       color='#2C3E50', linewidth=3, 
                       label='Ideal', alpha=0.9, zorder=5, linestyle='-')
                
                # Plot each method with enhanced styling
                for method, data in st.session_state.models_trained.items():
                    empirical_coverage = []
                    
                    progress = st.progress(0)
                    for i, a in enumerate(alpha_levels):
                        lower, upper = get_prediction_bounds(
                            method, data['models'], X_test, data['cal_scores'], a
                        )
                        cov = compute_coverage(Y_test, lower, upper)
                        empirical_coverage.append(cov)
                        progress.progress((i + 1) / len(alpha_levels))
                    
                    colors = COLOR_PALETTE[method]
                    linestyles = {'CQR': '--', 'AER': '-.', 'STD': ':'}
                    
                    ax.plot(1 - alpha_levels, empirical_coverage, 
                           color=colors['main'],
                           linestyle=linestyles.get(method, '-'),
                           linewidth=3, label=method, alpha=0.85, 
                           marker='o', markersize=5, markevery=3)
                
                ax.set_xlabel('1 - α (Target Coverage)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Empirical Coverage', fontsize=13, fontweight='bold')
                ax.set_title('Calibration Curves for Different Methods', 
                           fontsize=14, fontweight='bold', pad=15)
                ax.legend(fontsize=11, framealpha=0.95, edgecolor='#CCCCCC', 
                         shadow=True, loc='lower right')
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, 
                       color=COLOR_PALETTE['grid'])
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_facecolor(COLOR_PALETTE['background'])
                
                # Improve spine visibility
                for spine in ax.spines.values():
                    spine.set_edgecolor('#CCCCCC')
                    spine.set_linewidth(1.2)
                
                # Add diagonal reference line annotation
                ax.text(0.5, 0.45, 'Perfect Calibration', 
                       rotation=45, fontsize=10, alpha=0.5,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        else:
            st.info("👈 Please generate data and train models first!")


if __name__ == "__main__":
    main()