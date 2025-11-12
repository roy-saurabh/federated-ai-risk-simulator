"""
Federated AI Risk Simulator
===========================

A comprehensive Streamlit application for simulating federated learning
scenarios with AI risk assessment metrics. This tool allows data scientists
to explore the trade-offs between privacy, fairness, and utility in
decentralized AI systems.

Features:
- Interactive parameter configuration
- Real-time simulation with risk metrics
- Visualizations for accuracy, fairness, and privacy
- Download capabilities for presentations
- Comprehensive risk assessment dashboard

Usage:
    streamlit run trustworthy_ai_risk_simulator.py

Author: AffectLog Team
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from typing import List, Tuple, Dict, Any
import time


# Page configuration
st.set_page_config(
    page_title="Federated AI Risk Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    :root {
        --brand-primary: #004C8B;
        --brand-accent: #00AEEF;
        --brand-bg: #0b1f33;
    }
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #001e36, #0b3a66);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #f0f4f8;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .brand-left img {
        height: 28px;
        width: auto;
    }
    .brand-cta a {
        background: var(--brand-accent);
        color: #00111f !important;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 600;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .brand-cta a:hover {
        filter: brightness(0.95);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #004C8B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #004C8B;
        margin: 0.5rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }
    .risk-high {
        border-left-color: #D62728;
    }
    .risk-medium {
        border-left-color: #FF7F0E;
    }
    .risk-low {
        border-left-color: #2CA02C;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
        background-color: #004C8B;
        border: 1px solid rgba(0,0,0,0.05);
        color: #fff;
    }
    .stButton > button:hover {
        background-color: #0a66ad;
    }
</style>
""", unsafe_allow_html=True)

def _load_brand_logo_data_uri() -> str:
    """Try to load AffectLog logo from assets or session and return data URI; else empty string."""
    try:
        import os
        # Prefer uploaded logo in session
        if 'affectlog_logo_bytes' in st.session_state and st.session_state.affectlog_logo_bytes:
            b64 = base64.b64encode(st.session_state.affectlog_logo_bytes).decode()
            return f"data:image/png;base64,{b64}"
        # Fallback to local asset if exists
        assets_path = os.path.join(os.path.dirname(__file__), "assets", "affectlog_logo.png")
        if os.path.exists(assets_path):
            with open(assets_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{b64}"
    except Exception:
        pass
    return ""


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid function."""
    return 1 / (1 + np.exp(-z))


def generate_synthetic_data(n_clients: int, n_samples: int, dim: int = 2, seed: int = 42):
    """
    Generate synthetic binary classification data split across clients.
    
    Returns:
        client_data: List of (X_i, y_i) tuples for each client
        test_data: (X_test, y_test) tuple for held-out test data
    """
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=(dim,))
    true_b = rng.normal()
    X = rng.normal(size=(n_samples, dim))
    probs = sigmoid(X @ true_w + true_b)
    y = (probs > 0.5).astype(np.float32)
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    client_data = []
    sizes = np.full(n_clients, len(X_train) // n_clients, dtype=int)
    sizes[: len(X_train) % n_clients] += 1
    start = 0
    for sz in sizes:
        client_data.append((X_train[start : start + sz], y_train[start : start + sz]))
        start += sz
    return client_data, (X_test, y_test)


def compute_gradients(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute gradients for logistic regression."""
    preds = sigmoid(X @ w)
    error = preds - y
    grad = X.T @ error / len(X)
    return grad


def train_centralised(X: np.ndarray, y: np.ndarray, rounds: int = 20, lr: float = 0.2) -> np.ndarray:
    """Train a centralized logistic regression model."""
    dim = X.shape[1]
    w = np.zeros(dim)
    for _ in range(rounds):
        grad = compute_gradients(w, X, y)
        w -= lr * grad
    return w


def evaluate_model(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate model accuracy."""
    preds = (sigmoid(X @ w) > 0.5).astype(np.float32)
    return (preds == y).mean()


def run_federated_simulation(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray],
    rounds: int = 20,
    lr: float = 0.2,
    noise: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run federated learning simulation with comprehensive metrics collection.
    
    Returns:
        Dictionary containing all simulation results and metrics
    """
    rng = np.random.default_rng(seed)
    dim = client_data[0][0].shape[1]
    w_global = np.zeros(dim)
    
    # Initialize tracking variables
    accuracy_history = []
    fairness_disparity_history = []
    gradient_norms_history = []
    client_accuracies_history = []
    
    X_test, y_test = test_data
    
    for round_num in range(rounds):
        # Collect gradients from all clients
        grads = []
        norms = []
        client_accuracies = []
        
        for X_i, y_i in client_data:
            grad = compute_gradients(w_global, X_i, y_i)
            if noise > 0:
                grad += rng.normal(scale=noise, size=grad.shape)
            # Gradient magnitudes are a proxy for potential information leakage risk (Abadi et al., 2016)
            norms.append(np.linalg.norm(grad))
            grads.append(grad)
            
            # Compute client-specific accuracy
            client_acc = evaluate_model(w_global, X_i, y_i)
            client_accuracies.append(client_acc)
        
        # Update global model
        mean_grad = np.mean(grads, axis=0)
        w_global -= lr * mean_grad
        
        # Compute metrics for this round
        # Convergence of global model accuracy over test set (McMahan et al., 2017)
        global_acc = evaluate_model(w_global, X_test, y_test)
        # Inter-client disparity: max - min accuracy across clients (Mitchell et al., 2021)
        fairness_disparity = max(client_accuracies) - min(client_accuracies)
        
        # Store history
        accuracy_history.append(global_acc)
        fairness_disparity_history.append(fairness_disparity)
        gradient_norms_history.append(norms)
        client_accuracies_history.append(client_accuracies)
    
    # Train centralized model for comparison
    X_full = np.vstack([X for X, _ in client_data])
    y_full = np.hstack([y for _, y in client_data])
    w_central = train_centralised(X_full, y_full, rounds=rounds, lr=lr)
    central_acc = evaluate_model(w_central, X_test, y_test)
    
    # Compute final metrics
    final_fed_acc = accuracy_history[-1]
    final_disparity = fairness_disparity_history[-1]
    all_gradient_norms = np.array(gradient_norms_history)
    mean_grad_norm = all_gradient_norms.mean()
    std_grad_norm = all_gradient_norms.std()
    
    return {
        'w_global': w_global,
        'w_central': w_central,
        'accuracy_history': accuracy_history,
        'fairness_disparity_history': fairness_disparity_history,
        'gradient_norms_history': gradient_norms_history,
        'client_accuracies_history': client_accuracies_history,
        'central_accuracy': central_acc,
        'federated_accuracy': final_fed_acc,
        'final_disparity': final_disparity,
        'mean_gradient_norm': mean_grad_norm,
        'std_gradient_norm': std_grad_norm,
        'accuracy_difference': central_acc - final_fed_acc,
        # Additional metadata for research-grade visuals
        'n_test': len(y_test),
        'rounds': rounds,
        'noise': noise,
        'n_clients': len(client_data)
    }


def create_accuracy_plot(results: Dict[str, Any]) -> go.Figure:
    """Create research-grade accuracy comparison plot with CI band and trend."""
    rounds = list(range(1, len(results['accuracy_history']) + 1))
    
    acc = np.array(results['accuracy_history'])
    n_test = results.get('n_test', None)
    ci_lower = None
    ci_upper = None
    if n_test and n_test > 0:
        se = np.sqrt(np.clip(acc * (1 - acc) / n_test, 1e-9, None))
        ci_lower = np.clip(acc - 1.96 * se, 0, 1)
        ci_upper = np.clip(acc + 1.96 * se, 0, 1)
    
    # Regression fit (convergence rate)
    coeffs = np.polyfit(rounds, acc, deg=1)
    trend = np.poly1d(coeffs)(rounds)
    slope = coeffs[0]
    
    fig = go.Figure()
    
    # CI band
    if ci_lower is not None:
        fig.add_trace(go.Scatter(
            x=rounds,
            y=ci_lower,
            mode='lines',
            line=dict(color='rgba(0,174,239,0.0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=rounds,
            y=ci_upper,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,174,239,0.15)',
            line=dict(color='rgba(0,174,239,0.0)'),
            name='95% CI',
            hovertemplate='Round %{x}<br>95% CI: [%{customdata[0]:.3f}, %{y:.3f}]<extra></extra>',
            customdata=np.stack([ci_lower], axis=-1)
        ))
    
    # Federated accuracy line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=acc,
        mode='lines+markers',
        name='Federated Model',
        line=dict(color='#004C8B', width=3),
        marker=dict(size=6, color='#004C8B'),
        hovertemplate='Round %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=trend,
        mode='lines',
        name='Trend (polyfit)',
        line=dict(color='#00AEEF', width=2, dash='dash'),
        hovertemplate='Trend: %{y:.3f}<extra></extra>'
    ))
    
    # Centralized accuracy (baseline)
    fig.add_hline(
        y=results['central_accuracy'],
        line_dash="dot",
        line_color="#00AEEF",
        annotation_text=f"Centralized = {results['central_accuracy']:.3f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Convergence Dynamics in Federated Optimization",
        xaxis_title="Federated Round",
        yaxis_title="Global Accuracy",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.add_annotation(
        x=rounds[-1],
        y=trend[-1],
        text=f"Convergence rate (slope): {slope:.4f}",
        showarrow=False,
        font=dict(color='#004C8B')
    )
    
    return fig


def create_client_accuracy_evolution_plot(results: Dict[str, Any]) -> go.Figure:
    """Plot per-client accuracies across rounds using Plotly (avoids Vega-Lite warnings)."""
    client_hist = results.get('client_accuracies_history', [])
    if not client_hist:
        return go.Figure()
    rounds = list(range(1, len(client_hist) + 1))
    # client_hist is list over rounds, each entry is list over clients
    num_clients = len(client_hist[0])
    fig = go.Figure()
    for client_idx in range(num_clients):
        y_values = [client_hist[r][client_idx] for r in range(len(client_hist))]
        fig.add_trace(go.Scatter(
            x=rounds,
            y=y_values,
            mode='lines+markers',
            name=f'Client {client_idx + 1}',
            line=dict(width=2),
            hovertemplate='Round %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
        ))
    fig.update_layout(
        title="Per-Client Accuracy Evolution",
        xaxis_title="Federated Round",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def create_fairness_plot(results: Dict[str, Any]) -> go.Figure:
    """Create animated fairness disparity plot with per-round client accuracy distributions."""
    rounds = list(range(1, len(results['fairness_disparity_history']) + 1))
    disparities = results['fairness_disparity_history']
    client_hist = results['client_accuracies_history']
    
    # Regression on disparity
    coeffs = np.polyfit(rounds, disparities, deg=1)
    trend = np.poly1d(coeffs)(rounds)
    
    fig = go.Figure()
    
    # Disparity line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=disparities,
        mode='lines+markers',
        name='Disparity (max - min)',
        line=dict(color='#D62728', width=3),
        marker=dict(size=6),
        hovertemplate='Round %{x}<br>Disparity: %{y:.3f}<extra></extra>'
    ))
    # Trend line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=trend,
        mode='lines',
        name='Trend (polyfit)',
        line=dict(color='#FF7F0E', width=2, dash='dash'),
        hovertemplate='Trend: %{y:.3f}<extra></extra>'
    ))
    # Initial violin for round 1
    fig.add_trace(go.Violin(
        x=np.repeat([1], len(client_hist[0])),
        y=client_hist[0],
        name='Per-client Accuracy',
        line_color='#004C8B',
        fillcolor='rgba(0,76,139,0.1)',
        meanline_visible=True,
        points='all',
        hovertemplate='Client Accuracy: %{y:.3f}<extra></extra>'
    ))
    
    # Frames for animation
    frames = []
    for i, accs in enumerate(client_hist):
        frame = go.Frame(
            data=[
                go.Violin(
                    x=np.repeat([i + 1], len(accs)),
                    y=accs,
                    name='Per-client Accuracy',
                    line_color='#004C8B',
                    fillcolor='rgba(0,76,139,0.1)',
                    meanline_visible=True,
                    points='all',
                    hovertemplate='Client Accuracy: %{y:.3f}<extra></extra>'
                )
            ],
            traces=[2],  # update only the violin trace (index 2)
            name=str(i + 1)
        )
        frames.append(frame)
    
    fig.frames = frames
    fig.update_layout(
        title="Evolution of Inter-Client Fairness Disparity",
        xaxis_title="Federated Round",
        yaxis_title="Accuracy / Disparity",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": 1.15,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 400, "redraw": True}, "transition": {"duration": 200}}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]}
            ]
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Round: "},
            "pad": {"t": 50},
            "steps": [
                {"label": str(r), "method": "animate", "args": [[str(r)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for r in rounds
            ]
        }]
    )
    
    # Annotation for minimum disparity
    min_idx = int(np.argmin(disparities))
    fig.add_annotation(
        x=rounds[min_idx],
        y=disparities[min_idx],
        text=f"Min disparity: {disparities[min_idx]:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(color='#D62728')
    )
    
    return fig


def create_gradient_norms_plot(results: Dict[str, Any]) -> go.Figure:
    """Create gradient norms distribution plot with research-grade enhancements."""
    rounds = list(range(1, len(results['gradient_norms_history']) + 1))
    
    # Prepare data for box plot
    df_data = []
    for i, norms in enumerate(results['gradient_norms_history']):
        for norm in norms:
            df_data.append({'Round': i + 1, 'Gradient Norm': norm})
    
    df = pd.DataFrame(df_data)
    
    fig = px.box(
        df,
        x='Round',
        y='Gradient Norm',
        title="Empirical Distribution of Gradient Magnitudes (Privacy Proxy)",
        color_discrete_sequence=['#004C8B']
    )
    
    # Add mean gradient norm per round
    mean_per_round = df.groupby('Round')['Gradient Norm'].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=mean_per_round['Round'],
        y=mean_per_round['Gradient Norm'],
        mode='lines+markers',
        name='Mean Gradient Norm',
        line=dict(color='#00AEEF', width=2),
        marker=dict(size=6),
        hovertemplate='Round %{x}<br>Mean Norm: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Federated Round",
        yaxis_title="L2 Norm of Gradients",
        template='plotly_white',
        hovermode='x unified',
        height=450
    )
    
    return fig


def create_noise_vs_accuracy_plot(noise_results: List[Dict[str, Any]]) -> go.Figure:
    """Create noise vs accuracy trade-off plot."""
    noise_levels = [result['noise'] for result in noise_results]
    accuracies = [result['federated_accuracy'] for result in noise_results]
    disparities = [result['final_disparity'] for result in noise_results]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Noise vs Accuracy', 'Noise vs Fairness Disparity'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=noise_levels,
            y=accuracies,
            mode='lines+markers',
            name='Federated Accuracy',
            line=dict(color='#004C8B', width=3),
            hovertemplate='Noise σ: %{x:.2f}<br>Accuracy: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Disparity plot
    fig.add_trace(
        go.Scatter(
            x=noise_levels,
            y=disparities,
            mode='lines+markers',
            name='Fairness Disparity',
            line=dict(color='#D62728', width=3),
            hovertemplate='Noise σ: %{x:.2f}<br>Disparity: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Differential Privacy Noise (σ)", row=1, col=1)
    fig.update_xaxes(title_text="Differential Privacy Noise (σ)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Fairness Disparity", row=1, col=2)
    
    fig.update_layout(
        title="Privacy–Utility Trade-off Analysis",
        template='plotly_white',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='right', x=1)
    )
    
    return fig


def create_noise_client_accuracy_surface(
    n_rounds: int,
    learning_rate: float,
    base_seed: int
) -> go.Figure:
    """3D surface of Accuracy over Noise (σ) and Client Count using quick grid."""
    # Define a modest grid to keep compute practical
    noise_grid = np.round(np.linspace(0.0, 0.3, 7), 2)  # 0.00 to 0.30 step 0.05
    client_grid = np.array([2, 4, 8, 12, 16, 20])
    
    Z = np.zeros((len(client_grid), len(noise_grid)))
    
    for i, n_clients in enumerate(client_grid):
        # Regenerate data per client count for fairness
        client_data, test_data = generate_synthetic_data(
            n_clients=int(n_clients),
            n_samples=1000,
            seed=base_seed
        )
        for j, sigma in enumerate(noise_grid):
            res = run_federated_simulation(
                client_data=client_data,
                test_data=test_data,
                rounds=n_rounds,
                lr=learning_rate,
                noise=float(sigma),
                seed=base_seed
            )
            Z[i, j] = res['federated_accuracy']
    
    fig = go.Figure(data=[go.Surface(
        x=noise_grid,
        y=client_grid,
        z=Z,
        colorscale='Blues',
        colorbar=dict(title='Accuracy'),
        hovertemplate='σ: %{x:.2f}<br>Clients: %{y}<br>Accuracy: %{z:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Privacy–Utility–Scalability Frontier",
        scene=dict(
            xaxis_title="Noise (σ)",
            yaxis_title="Client Count",
            zaxis_title="Accuracy",
            zaxis=dict(range=[0, 1])
        ),
        template='plotly_white',
        height=600
    )
    return fig


def compute_noise_client_accuracy_grid(
    n_rounds: int,
    learning_rate: float,
    base_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute grid Z of accuracy for (noise, clients)."""
    noise_grid = np.round(np.linspace(0.0, 0.3, 7), 2)
    client_grid = np.array([2, 4, 8, 12, 16, 20])
    Z = np.zeros((len(client_grid), len(noise_grid)))
    for i, n_clients in enumerate(client_grid):
        client_data, test_data = generate_synthetic_data(
            n_clients=int(n_clients),
            n_samples=1000,
            seed=base_seed
        )
        for j, sigma in enumerate(noise_grid):
            res = run_federated_simulation(
                client_data=client_data,
                test_data=test_data,
                rounds=n_rounds,
                lr=learning_rate,
                noise=float(sigma),
                seed=base_seed
            )
            Z[i, j] = res['federated_accuracy']
    return noise_grid, client_grid, Z


def create_noise_client_accuracy_heatmap(
    noise_grid: np.ndarray,
    client_grid: np.ndarray,
    Z: np.ndarray
) -> go.Figure:
    """2D heatmap fallback for surface data."""
    fig = go.Figure(data=go.Heatmap(
        x=noise_grid,
        y=client_grid,
        z=Z,
        colorscale='Blues',
        colorbar=dict(title='Accuracy'),
        hovertemplate='σ: %{x:.2f}<br>Clients: %{y}<br>Accuracy: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title="Privacy–Utility–Scalability Frontier (2D Heatmap Fallback)",
        xaxis_title="Noise (σ)",
        yaxis_title="Client Count",
        yaxis=dict(type='linear', autorange='reversed'),  # Keep low clients at top visually if desired
        template='plotly_white',
        height=520
    )
    return fig


def get_risk_level(metric: float, metric_type: str) -> str:
    """Determine risk level based on metric value."""
    if metric_type == "disparity":
        if metric > 0.2:
            return "high"
        elif metric > 0.1:
            return "medium"
        else:
            return "low"
    elif metric_type == "gradient_norm":
        if metric > 0.5:
            return "high"
        elif metric > 0.2:
            return "medium"
        else:
            return "low"
    elif metric_type == "accuracy_diff":
        if abs(metric) > 0.1:
            return "high"
        elif abs(metric) > 0.05:
            return "medium"
        else:
            return "low"
    return "low"


def download_plot_as_png(fig: go.Figure, filename: str) -> str:
    """Convert plotly figure to downloadable PNG."""
    try:
        img_bytes = fig.to_image(format="png", width=800, height=600)
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename}</a>'
        return href
    except Exception as e:
        # If Chrome/Kaleido is not available, return a message instead
        return f'<small>💡 Download not available in cloud environment. Use browser screenshot for saving plots.</small>'


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient as a proxy for representational inequality."""
    v = np.array(values, dtype=float).flatten()
    if np.amin(v) < 0:
        v = v - np.amin(v)
    mean_v = np.mean(v) + 1e-12
    v_sorted = np.sort(v)
    n = len(v_sorted)
    cumulative = np.cumsum(v_sorted)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(np.clip(gini, 0.0, 1.0))


def create_radar_summary(results: Dict[str, Any], noise_level: float) -> go.Figure:
    """Radar chart comparing Baseline vs Current on normalized [0,1] metrics."""
    # Current metrics
    acc_curr = float(results['federated_accuracy'])
    fairness_curr = 1.0 - float(np.clip(results['final_disparity'], 0, 1))  # higher is better
    grad_hist = np.array(results['gradient_norms_history'])
    mean_grad = float(results['mean_gradient_norm'])
    max_grad = float(np.max(grad_hist))
    grad_score_curr = float(np.clip(1.0 - (mean_grad / (max_grad + 1e-9)), 0.0, 1.0))
    priv_curr = float(np.clip(noise_level / 0.5, 0.0, 1.0))
    
    # Baseline (centralized idealized)
    acc_base = float(results['central_accuracy'])
    fairness_base = 1.0
    grad_score_base = 1.0
    priv_base = 0.0
    
    categories = ['Accuracy', 'Fairness', 'Gradient Norm', 'Privacy Noise']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[acc_base, fairness_base, grad_score_base, priv_base],
        theta=categories,
        fill='toself',
        name='Baseline',
        line=dict(color='#00AEEF'),
        hovertemplate='%{theta}: %{r:.3f}<extra>Baseline</extra>'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[acc_curr, fairness_curr, grad_score_curr, priv_curr],
        theta=categories,
        fill='toself',
        name='Current',
        line=dict(color='#004C8B'),
        hovertemplate='%{theta}: %{r:.3f}<extra>Current</extra>'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Risk–Utility Summary (Radar)",
        template='plotly_white',
        height=520,
        legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='right', x=1)
    )
    return fig


def main():
    # Branding header with logo and CTA
    logo_uri = _load_brand_logo_data_uri()
    left_html = f"<img src='{logo_uri}' alt='AffectLog360°' />" if logo_uri else "AffectLog360°"
    st.markdown(f"""
    <div class="brand-bar">
        <div class="brand-left">
            {left_html}
            <span>Trustworthy AI for Digital Citizens</span>
        </div>
        <div class="brand-cta">
            <a href="https://app.affectlog.com" target="_blank" rel="noopener noreferrer">Open AffectLog 360°</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Header
    st.markdown('<h1 class="main-header">🤖 Federated AI Risk Simulator</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Simulation Parameters")
    # Sidebar branding + uploader
    if logo_uri:
        st.sidebar.image(logo_uri, caption="AffectLog 360°", width='stretch')
    uploaded = st.sidebar.file_uploader("Optional: Upload AffectLog logo (PNG)", type=["png"], accept_multiple_files=False)
    if uploaded is not None:
        st.session_state.affectlog_logo_bytes = uploaded.read()
        try:
            st.rerun()
        except Exception:
            # Fallback for older Streamlit versions
            pass
    st.sidebar.markdown(
        '<a href="https://app.affectlog.com" target="_blank">app.affectlog.com</a>',
        unsafe_allow_html=True
    )
    
    # Parameter inputs
    n_clients = st.sidebar.slider("Number of Clients", 2, 20, 8, help="Number of federated learning clients")
    n_samples = st.sidebar.slider("Total Samples", 100, 10000, 1000, step=100, help="Total number of training samples across all clients")
    n_rounds = st.sidebar.slider("Federated Rounds", 1, 50, 25, help="Number of federated training rounds")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.2, step=0.01, help="Learning rate for gradient descent")
    noise_level = st.sidebar.slider("DP Noise (σ)", 0.0, 0.5, 0.1, step=0.01, help="Differential privacy noise level")
    random_seed = st.sidebar.number_input("Random Seed", value=42, help="Random seed for reproducibility")
    show_advanced = st.sidebar.checkbox("Show Advanced Metrics", value=False)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running federated learning simulation..."):
            # Generate data
            client_data, test_data = generate_synthetic_data(
                n_clients=n_clients, 
                n_samples=n_samples, 
                seed=random_seed
            )
            
            # Run simulation
            results = run_federated_simulation(
                client_data=client_data,
                test_data=test_data,
                rounds=n_rounds,
                lr=learning_rate,
                noise=noise_level,
                seed=random_seed
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.client_data = client_data
            st.session_state.test_data = test_data
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Summary metrics
        st.header("📊 Risk Assessment Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_level = get_risk_level(results['accuracy_difference'], "accuracy_diff")
            st.markdown(f"""
            <div class="metric-card risk-{risk_level}">
                <h4>🎯 Centralized Accuracy</h4>
                <h2>{results['central_accuracy']:.3f}</h2>
                <small>Baseline performance</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_level = get_risk_level(results['accuracy_difference'], "accuracy_diff")
            st.markdown(f"""
            <div class="metric-card risk-{risk_level}">
                <h4>🤝 Federated Accuracy</h4>
                <h2>{results['federated_accuracy']:.3f}</h2>
                <small>Privacy-preserving model</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            risk_level = get_risk_level(results['final_disparity'], "disparity")
            st.markdown(f"""
            <div class="metric-card risk-{risk_level}">
                <h4>⚖️ Fairness Disparity</h4>
                <h2>{results['final_disparity']:.3f}</h2>
                <small>Max - Min client accuracy</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            risk_level = get_risk_level(results['mean_gradient_norm'], "gradient_norm")
            st.markdown(f"""
            <div class="metric-card risk-{risk_level}">
                <h4>🔒 Avg Gradient Norm</h4>
                <h2>{results['mean_gradient_norm']:.3f}</h2>
                <small>±{results['std_gradient_norm']:.3f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        if show_advanced:
            # Advanced metrics panel
            st.subheader("🧪 Advanced Metrics")
            colA, colB, colC = st.columns(3)
            # MAE between federated per-round and centralized accuracy
            mae = float(np.mean(np.abs(np.array(results['accuracy_history']) - results['central_accuracy'])))
            # Gini on final per-client accuracies
            gini = gini_coefficient(results['client_accuracies_history'][-1]) if results.get('client_accuracies_history') else 0.0
            # Privacy loss epsilon ≈ 1/(2σ²)
            eps = (1.0 / (2.0 * (noise_level ** 2))) if noise_level > 0 else float('inf')
            with colA:
                st.metric("MAE (Fed vs Central)", f"{mae:.3f}")
            with colB:
                st.metric("Gini (Client Accuracy)", f"{gini:.3f}")
            with colC:
                st.metric("Privacy Loss ε (approx.)", f"{'∞' if np.isinf(eps) else f'{eps:.2f}'}")
        
        # Risk interpretation
        st.subheader("🔍 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Accuracy Gap**: {results['accuracy_difference']:.3f}
            - {'⚠️ High risk' if abs(results['accuracy_difference']) > 0.1 else '✅ Low risk'}
            - {'Significant performance degradation in federated setting' if abs(results['accuracy_difference']) > 0.1 else 'Minimal performance impact from federation'}
            """)
            
            st.info(f"""
            **Fairness Disparity**: {results['final_disparity']:.3f}
            - {'⚠️ High risk' if results['final_disparity'] > 0.2 else '✅ Low risk' if results['final_disparity'] < 0.1 else '⚠️ Medium risk'}
            - {'Model shows bias toward certain clients' if results['final_disparity'] > 0.2 else 'Model performs fairly across clients' if results['final_disparity'] < 0.1 else 'Moderate bias detected'}
            """)
        
        with col2:
            st.info(f"""
            **Privacy Signal**: {results['mean_gradient_norm']:.3f}
            - {'⚠️ High risk' if results['mean_gradient_norm'] > 0.5 else '✅ Low risk' if results['mean_gradient_norm'] < 0.2 else '⚠️ Medium risk'}
            - {'Large gradients may leak information' if results['mean_gradient_norm'] > 0.5 else 'Small gradients reduce information leakage' if results['mean_gradient_norm'] < 0.2 else 'Moderate information leakage risk'}
            """)
            
            st.info(f"""
            **Differential Privacy**: σ = {noise_level}
            - {'✅ Strong privacy protection' if noise_level > 0.3 else '⚠️ Moderate protection' if noise_level > 0.1 else '❌ Weak privacy protection'}
            - {'High noise reduces utility but increases privacy' if noise_level > 0.3 else 'Balanced privacy-utility trade-off' if noise_level > 0.1 else 'Low noise may not provide sufficient privacy'}
            """)
        
        # Visualizations
        st.header("📈 Performance & Risk Metrics")
        
        # Create plots
        accuracy_fig = create_accuracy_plot(results)
        fairness_fig = create_fairness_plot(results)
        gradient_fig = create_gradient_norms_plot(results)
        
        # Display plots in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(accuracy_fig, width='stretch')
            st.markdown(download_plot_as_png(accuracy_fig, "accuracy_over_rounds"), unsafe_allow_html=True)
            with st.expander("Technical Notes"):
                st.markdown("- Gradient-based federated optimization convergence follows FedAvg principles (McMahan et al., 2017).")
                st.markdown("- Confidence bands use normal approximation of binomial accuracy on test set.")
        
        with col2:
            st.plotly_chart(fairness_fig, width='stretch')
            st.markdown(download_plot_as_png(fairness_fig, "fairness_disparity"), unsafe_allow_html=True)
            with st.expander("Technical Notes"):
                st.markdown("- Inter-client disparity uses max–min accuracy; per-round violins show distribution across clients (Mitchell et al., 2021).")
        
        # Gradient norms plot (full width)
        st.plotly_chart(gradient_fig, width='stretch')
        st.markdown(download_plot_as_png(gradient_fig, "gradient_norms_distribution"), unsafe_allow_html=True)
        with st.expander("Technical Notes"):
            st.markdown("- Gradient magnitudes approximate expected information leakage per update; DP adds calibrated noise (Abadi et al., 2016).")
        
        # Optional: Noise vs Accuracy analysis
        st.header("🔬 Privacy–Utility Trade-off Analysis")
        
        if st.button("Analyze Noise Impact"):
            with st.spinner("Running noise sensitivity analysis..."):
                noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                noise_results = []
                
                for noise in noise_levels:
                    result = run_federated_simulation(
                        client_data=st.session_state.client_data,
                        test_data=st.session_state.test_data,
                        rounds=n_rounds,
                        lr=learning_rate,
                        noise=noise,
                        seed=random_seed
                    )
                    result['noise'] = noise
                    noise_results.append(result)
                
                noise_fig = create_noise_vs_accuracy_plot(noise_results)
                st.plotly_chart(noise_fig, width='stretch')
                st.markdown(download_plot_as_png(noise_fig, "privacy_utility_tradeoff"), unsafe_allow_html=True)
                with st.expander("Technical Notes"):
                    st.markdown("- Increasing σ improves privacy but typically reduces utility; visualize joint response across metrics.")
        
        # 3D surface: Noise (σ) × Client Count → Accuracy
        st.header("🌐 Privacy–Utility–Scalability Frontier (3D)")
        render_mode = st.selectbox(
            "Rendering mode (choose fallback if WebGL is unavailable in your browser):",
            ["Interactive 3D (WebGL)", "Static 3D Image (PNG)", "2D Heatmap (Fallback)"],
            index=0
        )
        if st.button("Generate 3D Surface"):
            with st.spinner("Computing surface (σ, clients) → accuracy..."):
                noise_grid, client_grid, Z = compute_noise_client_accuracy_grid(
                    n_rounds=n_rounds,
                    learning_rate=learning_rate,
                    base_seed=random_seed
                )
            if render_mode == "Interactive 3D (WebGL)":
                surface_fig = go.Figure(data=[go.Surface(
                    x=noise_grid, y=client_grid, z=Z,
                    colorscale='Blues',
                    colorbar=dict(title='Accuracy'),
                    hovertemplate='σ: %{x:.2f}<br>Clients: %{y}<br>Accuracy: %{z:.3f}<extra></extra>'
                )])
                surface_fig.update_layout(
                    title="Privacy–Utility–Scalability Frontier",
                    scene=dict(
                        xaxis_title="Noise (σ)",
                        yaxis_title="Client Count",
                        zaxis_title="Accuracy",
                        zaxis=dict(range=[0, 1])
                    ),
                    template='plotly_white',
                    height=600
                )
                st.plotly_chart(surface_fig, width='stretch')
                st.markdown(download_plot_as_png(surface_fig, "privacy_utility_scalability_surface"), unsafe_allow_html=True)
            elif render_mode == "Static 3D Image (PNG)":
                static_fig = go.Figure(data=[go.Surface(
                    x=noise_grid, y=client_grid, z=Z,
                    colorscale='Blues',
                    colorbar=dict(title='Accuracy'),
                )])
                static_fig.update_layout(
                    title="Privacy–Utility–Scalability Frontier (Static 3D)",
                    scene=dict(
                        xaxis_title="Noise (σ)",
                        yaxis_title="Client Count",
                        zaxis_title="Accuracy",
                        zaxis=dict(range=[0, 1])
                    ),
                    template='plotly_white',
                    height=600
                )
                try:
                    img_bytes = static_fig.to_image(format="png", width=1100, height=700)
                    st.image(img_bytes, caption="Static 3D Surface (PNG)", width='stretch')
                    st.markdown(download_plot_as_png(static_fig, "privacy_utility_scalability_surface_static"), unsafe_allow_html=True)
                except Exception:
                    # Fallback to heatmap if static rendering backend isn't available
                    heatmap_fig = create_noise_client_accuracy_heatmap(noise_grid, client_grid, Z)
                    st.plotly_chart(heatmap_fig, width='stretch')
                    st.info("Static image backend not available. Showing 2D heatmap as fallback.")
            else:
                heatmap_fig = create_noise_client_accuracy_heatmap(noise_grid, client_grid, Z)
                st.plotly_chart(heatmap_fig, width='stretch')
                st.markdown(download_plot_as_png(heatmap_fig, "privacy_utility_scalability_heatmap"), unsafe_allow_html=True)
            with st.expander("Technical Notes"):
                st.markdown("- Surface shows how accuracy changes with DP noise σ and number of clients; illustrates privacy–utility–scalability trade-offs.")
        
        # Detailed client analysis
        st.header("👥 Per-Client Analysis")
        
        if 'client_accuracies_history' in results:
            client_df = pd.DataFrame(results['client_accuracies_history']).T
            client_df.columns = [f'Client {i+1}' for i in range(len(client_df.columns))]
            client_df.index = [f'Round {i+1}' for i in range(len(client_df))]
            
            st.subheader("Client Accuracy Evolution")
            client_evo_fig = create_client_accuracy_evolution_plot(results)
            st.plotly_chart(client_evo_fig, width='stretch')
            
            # Final client accuracies
            final_accuracies = results['client_accuracies_history'][-1]
            client_summary = pd.DataFrame({
                'Client': [f'Client {i+1}' for i in range(len(final_accuracies))],
                'Final Accuracy': final_accuracies,
                'Risk Level': [get_risk_level(acc, "disparity") for acc in final_accuracies]
            })
            
            st.subheader("Final Client Performance")
            st.dataframe(client_summary, width='stretch')
        
        # Radar summary
        st.header("🧭 Multi-Objective Summary")
        radar_fig = create_radar_summary(results, noise_level=noise_level)
        st.plotly_chart(radar_fig, width='stretch')
        st.markdown(download_plot_as_png(radar_fig, "risk_utility_radar"), unsafe_allow_html=True)
        with st.expander("Technical Notes"):
            st.markdown("- Radar chart compares current run to centralized baseline across normalized axes (accuracy, fairness, gradient proxy, privacy noise).")
    
    # Instructions
    if 'results' not in st.session_state:
        st.info("""
        ### 🎯 How to Use This Simulator
        
        1. **Configure Parameters**: Use the sidebar to set your simulation parameters
        2. **Run Simulation**: Click "Run Simulation" to start the federated learning process
        3. **Analyze Results**: Review the risk assessment metrics and visualizations
        4. **Save Plots**: Use browser screenshot or download links (if available) to save charts for presentations
        5. **Experiment**: Try different parameters to explore privacy-utility trade-offs
        
        ### 📊 Understanding the Metrics
        
        - **Centralized vs Federated Accuracy**: Compare performance between traditional and privacy-preserving approaches
        - **Fairness Disparity**: Measures bias across clients (higher = more bias)
        - **Gradient Norms**: Indicates information leakage risk (higher = more risk)
        - **Differential Privacy Noise**: Balances privacy protection with model utility
        
        ### 💡 Note on Downloads
        - Download functionality may not be available in cloud environments
        - Use browser screenshot (Cmd/Ctrl + Shift + 4) to save plots for presentations
        - All visualizations are interactive and can be zoomed/explored
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Federated AI Risk Simulator | Built for AffectLog 360° Demo</p>
        <p>Explore the trade-offs between privacy, fairness, and utility in decentralized AI systems</p>
        <p>Part of the Council of Europe – Ministry of Slovenia Hands-On Risk & Rights Lab</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 
