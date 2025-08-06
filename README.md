# Federated AI Risk Simulator

A comprehensive, interactive web application for simulating federated learning scenarios with AI risk assessment metrics. This tool allows data scientists to explore the trade-offs between privacy, fairness, and utility in decentralized AI systems.

## 🎯 Features

- **Interactive Parameter Configuration**: Adjust number of clients, samples, rounds, learning rate, and differential privacy noise
- **Real-time Simulation**: Run federated learning simulations with comprehensive metrics collection
- **Risk Assessment Dashboard**: Visualize accuracy, fairness disparity, and privacy metrics
- **Download Capabilities**: Export plots as PNG files for presentations
- **Privacy-Utility Trade-off Analysis**: Explore how differential privacy affects model performance
- **Per-Client Analysis**: Monitor individual client performance and bias detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # If you have the files locally, navigate to the project directory
   cd /path/to/federated-ai-risk-simulator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run federated_ai_risk_simulator.py
   ```

4. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL shown in your terminal

## 📊 Understanding the Metrics

### Core Metrics

- **Centralized vs Federated Accuracy**: Compare performance between traditional and privacy-preserving approaches
- **Fairness Disparity**: Measures bias across clients (higher values indicate more bias)
- **Gradient Norms**: Indicates information leakage risk (higher values suggest greater privacy risk)
- **Differential Privacy Noise**: Balances privacy protection with model utility

### Risk Levels

- **🟢 Low Risk**: Minimal concerns about bias, privacy, or performance
- **🟡 Medium Risk**: Moderate concerns that should be monitored
- **🔴 High Risk**: Significant concerns requiring attention

## 🎛️ Parameter Guide

### Simulation Parameters

- **Number of Clients (2-20)**: Number of federated learning participants
- **Total Samples (100-10,000)**: Total training data across all clients
- **Federated Rounds (1-50)**: Number of training iterations
- **Learning Rate (0.01-1.0)**: Step size for gradient descent
- **DP Noise (σ) (0.0-0.5)**: Differential privacy noise level
- **Random Seed**: For reproducible results

### Recommended Starting Values

- **8 clients, 1000 samples, 25 rounds, 0.2 learning rate, 0.1 noise**
- This provides a good balance for exploring the privacy-utility trade-off

## 📈 Visualizations

### 1. Accuracy Over Rounds
- Shows federated model performance progression
- Includes centralized baseline for comparison
- Helps identify convergence patterns

### 2. Fairness Disparity Over Rounds
- Tracks bias evolution across clients
- Higher values indicate increasing bias
- Critical for detecting unfair model behavior

### 3. Gradient Norms Distribution
- Box plots showing gradient magnitude distribution
- Helps assess information leakage risk
- Lower norms generally indicate better privacy

### 4. Privacy-Utility Trade-off Analysis
- Optional analysis showing noise impact
- Demonstrates the fundamental trade-off
- Useful for policy decisions

## 🔬 Use Cases

### For Researchers
- Explore federated learning dynamics
- Study privacy-utility trade-offs
- Validate fairness metrics

### For Practitioners
- Assess deployment risks
- Optimize hyperparameters
- Communicate risks to stakeholders

### For Educators
- Demonstrate federated learning concepts
- Illustrate AI risk assessment
- Show privacy-preserving techniques

## 🎯 Example Workflow

1. **Configure Parameters**: Set 8 clients, 1000 samples, 25 rounds, 0.2 learning rate, 0.1 noise
2. **Run Simulation**: Click "Run Simulation" and wait for completion
3. **Review Summary**: Check the risk assessment cards for initial insights
4. **Analyze Plots**: Examine accuracy, fairness, and gradient norm visualizations
5. **Download Results**: Save plots for presentations or reports
6. **Experiment**: Try different parameters to explore trade-offs

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Pure Python with NumPy for computations
- **Visualization**: Plotly for interactive charts
- **Data**: Synthetic binary classification data

### Algorithm
- **Model**: Logistic regression with gradient descent
- **Federation**: FedAvg (Federated Averaging)
- **Privacy**: Gaussian differential privacy
- **Evaluation**: Accuracy on held-out test set

### Risk Metrics
- **Fairness Disparity**: max(client_accuracies) - min(client_accuracies)
- **Gradient Norm**: L2 norm of client gradients
- **Accuracy Gap**: centralized_accuracy - federated_accuracy

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run federated_ai_risk_simulator.py --server.port 8502
   ```

2. **Missing dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Slow performance**
   - Reduce number of samples or rounds
   - Use fewer clients for faster simulation

4. **Plot download issues**
   - Ensure kaleido is installed: `pip install kaleido`
   - Check browser download settings

### Performance Tips

- Start with smaller datasets for quick exploration
- Use fewer rounds for rapid iteration
- Increase noise levels gradually to see effects

## 📚 Further Reading

### Federated Learning
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)

### AI Risk Assessment
- [Towards Accountability for Machine Learning Datasets](https://arxiv.org/abs/2009.06077)
- [Fairness in Machine Learning](https://arxiv.org/abs/1706.02409)

### Differential Privacy
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

## 🤝 Contributing

This simulator is designed for the AffectLog 360° Demo and Workshop sessions. For questions or improvements, please contact the development team.

## 📄 License

This project is developed for educational and research purposes as part of the AffectLog initiative.

---

**Built for AffectLog 360° Demo** 🤖 | Explore the trade-offs between privacy, fairness, and utility in decentralized AI systems 