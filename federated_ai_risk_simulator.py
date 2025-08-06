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
    streamlit run federated_ai_risk_simulator.py

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high {
        border-left-color: #d62728;
    }
    .risk-medium {
        border-left-color: #ff7f0e;
    }
    .risk-low {
        border-left-color: #2ca02c;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


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
            norms.append(np.linalg.norm(grad))
            grads.append(grad)
            
            # Compute client-specific accuracy
            client_acc = evaluate_model(w_global, X_i, y_i)
            client_accuracies.append(client_acc)
        
        # Update global model
        mean_grad = np.mean(grads, axis=0)
        w_global -= lr * mean_grad
        
        # Compute metrics for this round
        global_acc = evaluate_model(w_global, X_test, y_test)
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
        'accuracy_difference': central_acc - final_fed_acc
    }


def create_accuracy_plot(results: Dict[str, Any]) -> go.Figure:
    """Create accuracy comparison plot."""
    rounds = list(range(1, len(results['accuracy_history']) + 1))
    
    fig = go.Figure()
    
    # Federated accuracy line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=results['accuracy_history'],
        mode='lines+markers',
        name='Federated Model',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Centralized accuracy line
    fig.add_hline(
        y=results['central_accuracy'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Centralized: {results['central_accuracy']:.3f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Accuracy Over Federated Rounds",
        xaxis_title="Federated Round",
        yaxis_title="Global Accuracy",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_fairness_plot(results: Dict[str, Any]) -> go.Figure:
    """Create fairness disparity plot."""
    rounds = list(range(1, len(results['fairness_disparity_history']) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rounds,
        y=results['fairness_disparity_history'],
        mode='lines+markers',
        name='Fairness Disparity',
        line=dict(color='#d62728', width=3),
        marker=dict(size=6),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Fairness Disparity Over Rounds",
        xaxis_title="Federated Round",
        yaxis_title="Disparity (Max - Min Client Accuracy)",
        yaxis=dict(range=[0, max(results['fairness_disparity_history']) * 1.1]),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_gradient_norms_plot(results: Dict[str, Any]) -> go.Figure:
    """Create gradient norms distribution plot."""
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
        title="Distribution of Client Gradient Norms by Round"
    )
    
    fig.update_layout(
        xaxis_title="Federated Round",
        yaxis_title="L2 Norm of Gradients",
        template='plotly_white'
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
            line=dict(color='#1f77b4', width=3)
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
            line=dict(color='#d62728', width=3)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Differential Privacy Noise (σ)", row=1, col=1)
    fig.update_xaxes(title_text="Differential Privacy Noise (σ)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Fairness Disparity", row=1, col=2)
    
    fig.update_layout(
        title="Privacy-Utility Trade-off Analysis",
        template='plotly_white',
        height=400
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


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Federated AI Risk Simulator</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Simulation Parameters")
    
    # Parameter inputs
    n_clients = st.sidebar.slider("Number of Clients", 2, 20, 8, help="Number of federated learning clients")
    n_samples = st.sidebar.slider("Total Samples", 100, 10000, 1000, step=100, help="Total number of training samples across all clients")
    n_rounds = st.sidebar.slider("Federated Rounds", 1, 50, 25, help="Number of federated training rounds")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.2, step=0.01, help="Learning rate for gradient descent")
    noise_level = st.sidebar.slider("DP Noise (σ)", 0.0, 0.5, 0.1, step=0.01, help="Differential privacy noise level")
    random_seed = st.sidebar.number_input("Random Seed", value=42, help="Random seed for reproducibility")
    
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
            st.plotly_chart(accuracy_fig, use_container_width=True)
            st.markdown(download_plot_as_png(accuracy_fig, "accuracy_over_rounds"), unsafe_allow_html=True)
        
        with col2:
            st.plotly_chart(fairness_fig, use_container_width=True)
            st.markdown(download_plot_as_png(fairness_fig, "fairness_disparity"), unsafe_allow_html=True)
        
        # Gradient norms plot (full width)
        st.plotly_chart(gradient_fig, use_container_width=True)
        st.markdown(download_plot_as_png(gradient_fig, "gradient_norms_distribution"), unsafe_allow_html=True)
        
        # Optional: Noise vs Accuracy analysis
        st.header("🔬 Privacy-Utility Trade-off Analysis")
        
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
                st.plotly_chart(noise_fig, use_container_width=True)
                st.markdown(download_plot_as_png(noise_fig, "privacy_utility_tradeoff"), unsafe_allow_html=True)
        
        # Detailed client analysis
        st.header("👥 Per-Client Analysis")
        
        if 'client_accuracies_history' in results:
            client_df = pd.DataFrame(results['client_accuracies_history']).T
            client_df.columns = [f'Client {i+1}' for i in range(len(client_df.columns))]
            client_df.index = [f'Round {i+1}' for i in range(len(client_df))]
            
            st.subheader("Client Accuracy Evolution")
            st.line_chart(client_df)
            
            # Final client accuracies
            final_accuracies = results['client_accuracies_history'][-1]
            client_summary = pd.DataFrame({
                'Client': [f'Client {i+1}' for i in range(len(final_accuracies))],
                'Final Accuracy': final_accuracies,
                'Risk Level': [get_risk_level(acc, "disparity") for acc in final_accuracies]
            })
            
            st.subheader("Final Client Performance")
            st.dataframe(client_summary, use_container_width=True)
    
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
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 