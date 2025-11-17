# Federated AI Risk Simulator

A comprehensive, interactive web application for simulating federated learning scenarios with AI risk assessment metrics. This tool allows data scientists to explore the trade-offs between privacy, fairness, and utility in decentralized AI systems.

## 🎯 Features

- **Interactive Parameter Configuration**: Adjust number of clients, samples, rounds, learning rate, and differential privacy noise
- **Real-time Simulation**: Run federated learning simulations with comprehensive metrics collection
- **Risk Assessment Dashboard**: Visualize accuracy, fairness disparity, and privacy metrics
- **Advanced Metrics**: Gini coefficient, MAE, privacy loss (ε), and gradient norm analysis
- **Download Capabilities**: Export plots as PNG files for presentations
- **Privacy-Utility Trade-off Analysis**: Explore how differential privacy affects model performance
- **Per-Client Analysis**: Monitor individual client performance and bias detection
- **3D Surface Visualization**: Explore privacy-utility-scalability frontiers
- **Multi-Objective Summary**: Radar chart comparing baseline vs. federated performance

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

#### 1. **Centralized vs Federated Accuracy**

Compares performance between traditional centralized learning and privacy-preserving federated approaches.

**Mathematical Formulation:**
- **Centralized Accuracy**: $A_{central} = \frac{1}{n_{test}} \sum_{i=1}^{n_{test}} \mathbb{1}[\hat{y}_i = y_i]$
- **Federated Accuracy**: $A_{fed} = \frac{1}{n_{test}} \sum_{i=1}^{n_{test}} \mathbb{1}[\hat{y}_i^{fed} = y_i]$
- **Accuracy Gap**: $\Delta A = A_{central} - A_{fed}$

Where $\hat{y}_i$ is the predicted label and $y_i$ is the true label.

**Scientific Rationale**: The accuracy gap quantifies the utility cost of privacy-preserving federated learning. Larger gaps indicate significant performance degradation, which may be unacceptable for certain applications (McMahan et al., 2017).

#### 2. **Fairness Disparity**

Measures bias across clients by computing the difference between maximum and minimum client accuracies.

**Mathematical Formulation:**

$$
D_{fairness} = \max_{k \in [K]} A_k - \min_{k \in [K]} A_k
$$

Where $A_k$ is the accuracy of client $k$ and $K$ is the total number of clients.

**Scientific Rationale**: This metric captures inter-client performance disparity, a key indicator of algorithmic fairness. High disparity suggests the model benefits some clients disproportionately, violating fairness principles (Mitchell et al., 2021). Values > 0.2 indicate high risk, 0.1-0.2 medium risk, and < 0.1 low risk.

#### 3. **Gradient Norms**

L2 norm of client gradients, serving as a proxy for information leakage risk.

**Mathematical Formulation:**

$$
\|\nabla_k\|_2 = \sqrt{\sum_{j=1}^{d} \left( \frac{\partial \mathcal{L}}{\partial w_j} \right)^2}
$$

Where $\nabla_k$ is the gradient vector for client $k$, $d$ is the model dimension, and $\mathcal{L}$ is the loss function.

**Scientific Rationale**: Larger gradient magnitudes contain more information about local data, increasing the risk of privacy leakage through gradient inversion attacks (Abadi et al., 2016). The mean gradient norm across rounds provides an aggregate privacy risk measure.

#### 4. **Differential Privacy Noise**

Gaussian noise added to gradients for privacy protection, parameterized by standard deviation $\sigma$.

**Mathematical Formulation:**

$$
\tilde{\nabla}_k = \nabla_k + \mathcal{N}(0, \sigma^2 I)
$$

Where $\tilde{\nabla}_k$ is the noisy gradient and $\mathcal{N}(0, \sigma^2 I)$ is multivariate Gaussian noise.

**Privacy Loss (ε) Approximation:**

$$
\varepsilon \approx \frac{1}{2\sigma^2}
$$

**Scientific Rationale**: Differential privacy provides formal privacy guarantees. Higher $\sigma$ increases privacy (lower $\varepsilon$) but typically reduces model utility. The privacy-utility trade-off is fundamental in federated learning (Dwork & Roth, 2014).

### Advanced Metrics

#### 5. **Mean Absolute Error (MAE)**

Measures the average absolute difference between federated and centralized accuracies across rounds.

**Mathematical Formulation:**

$$
\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |A_{central} - A_{fed}^{(t)}|
$$

Where $T$ is the number of federated rounds and $A_{fed}^{(t)}$ is federated accuracy at round $t$.

**Scientific Rationale**: MAE quantifies the persistent deviation from centralized performance, helping identify convergence issues or systematic biases in federated optimization.

#### 6. **Gini Coefficient**

Measures inequality in client accuracy distribution, adapted from economics.

**Mathematical Formulation:**

$$
G = \frac{n + 1 - 2 \sum_{i=1}^{n} \frac{(n+1-i) \cdot A_{(i)}}{\sum_{j=1}^{n} A_j}}{n}
$$

Where $A_{(i)}$ are sorted client accuracies in ascending order and $n$ is the number of clients.

**Scientific Rationale**: The Gini coefficient (0 = perfect equality, 1 = maximum inequality) quantifies representational inequality across clients. Higher values indicate that some clients benefit significantly more than others, raising fairness concerns.

#### 7. **Privacy Loss (ε)**

Approximate differential privacy parameter indicating privacy guarantee strength.

**Mathematical Formulation:**

$$
\varepsilon = \begin{cases}
\frac{1}{2\sigma^2} & \text{if } \sigma > 0 \\
\infty & \text{if } \sigma = 0
\end{cases}
$$

**Scientific Rationale**: Lower $\varepsilon$ values indicate stronger privacy guarantees. Typically, $\varepsilon < 1$ is considered strong privacy, $1 \leq \varepsilon \leq 10$ moderate, and $\varepsilon > 10$ weak. This approximation assumes Gaussian mechanism with unit sensitivity.

### Risk Levels

- **🟢 Low Risk**: Minimal concerns about bias, privacy, or performance
  - Fairness Disparity < 0.1
  - Gradient Norm < 0.2
  - Accuracy Gap < 0.05
- **🟡 Medium Risk**: Moderate concerns that should be monitored
  - Fairness Disparity: 0.1-0.2
  - Gradient Norm: 0.2-0.5
  - Accuracy Gap: 0.05-0.1
- **🔴 High Risk**: Significant concerns requiring attention
  - Fairness Disparity > 0.2
  - Gradient Norm > 0.5
  - Accuracy Gap > 0.1

## 🎛️ Parameter Guide

### Simulation Parameters

- **Number of Clients (2-20)**: Number of federated learning participants
  - More clients increase heterogeneity but may slow convergence
- **Total Samples (100-10,000)**: Total training data across all clients
  - Larger datasets improve model quality but increase computation time
- **Federated Rounds (1-50)**: Number of training iterations
  - More rounds improve convergence but increase communication costs
- **Learning Rate (0.01-1.0)**: Step size for gradient descent
  - Higher rates converge faster but may overshoot optimal solution
- **DP Noise (σ) (0.0-0.5)**: Differential privacy noise level
  - Higher noise improves privacy but reduces model accuracy
- **Random Seed**: For reproducible results

### Recommended Starting Values

- **8 clients, 1000 samples, 25 rounds, 0.2 learning rate, 0.1 noise**
- This provides a good balance for exploring the privacy-utility trade-off

## 📈 Visualizations

### 1. **Convergence Dynamics in Federated Optimization**

**Description**: Shows federated model performance progression over rounds with statistical confidence bands and convergence trend analysis.

**Features**:
- Federated accuracy trajectory with markers
- 95% confidence intervals using normal approximation: $\text{CI} = A \pm 1.96 \cdot \sqrt{\frac{A(1-A)}{n_{test}}}$
- Linear trend line (polyfit) showing convergence rate: $A(t) = \alpha t + \beta$
- Centralized baseline for comparison
- Convergence rate annotation (slope coefficient)

**Scientific Value**: 
- Validates FedAvg convergence properties (McMahan et al., 2017)
- Confidence bands account for test set size uncertainty
- Trend analysis identifies convergence speed and stability
- Helps diagnose optimization issues (oscillations, slow convergence)

**Interpretation**:
- Steep positive slope: Rapid learning
- Flat slope: Convergence reached
- Negative slope: Overfitting or optimization instability
- Wide CI bands: High uncertainty (small test set)

### 2. **Per-Client Accuracy Evolution**

**Description**: Individual client accuracy trajectories across federated rounds, enabling identification of client-specific performance patterns.

**Features**:
- Separate line for each client
- Color-coded client identification
- Interactive hover tooltips

**Scientific Value**:
- Reveals client heterogeneity and data distribution skew
- Identifies clients with consistently poor performance (potential data quality issues)
- Detects convergence disparities (some clients benefit more than others)
- Essential for fairness analysis

**Interpretation**:
- Converging lines: Homogeneous client performance
- Diverging lines: High heterogeneity, potential fairness issues
- Flat lines: Clients not benefiting from federation
- Oscillating lines: Optimization instability

### 3. **Evolution of Inter-Client Fairness Disparity**

**Description**: Animated visualization showing fairness disparity evolution with per-round client accuracy distributions.

**Features**:
- Disparity line (max - min accuracy): $D(t) = \max_k A_k(t) - \min_k A_k(t)$
- Linear trend analysis
- Animated violin plots showing per-round client accuracy distributions
- Play/pause controls and round slider
- Minimum disparity annotation

**Scientific Value**:
- Tracks fairness evolution over training (Mitchell et al., 2021)
- Violin plots reveal distribution shape (skew, multimodality)
- Trend analysis identifies whether disparity increases or decreases
- Critical for detecting emergent bias during training

**Interpretation**:
- Decreasing disparity: Improving fairness
- Increasing disparity: Growing bias
- Stable disparity: Persistent fairness issues
- Wide violins: High client heterogeneity

### 4. **Empirical Distribution of Gradient Magnitudes (Privacy Proxy)**

**Description**: Box plots showing gradient norm distribution across clients per round, with mean trend overlay.

**Features**:
- Box plots per round (quartiles, outliers)
- Mean gradient norm trend line: $\bar{\|\nabla\|}(t) = \frac{1}{K} \sum_{k=1}^{K} \|\nabla_k^{(t)}\|_2$
- Standard deviation visualization through box plot spread

**Scientific Value**:
- Gradient norms proxy for information leakage risk (Abadi et al., 2016)
- Large gradients contain more data information, enabling gradient inversion attacks
- Distribution analysis reveals client heterogeneity in gradient magnitudes
- Mean trend shows whether privacy risk increases or decreases over training

**Interpretation**:
- Decreasing mean: Reducing privacy risk (gradients shrinking)
- Increasing mean: Growing privacy risk
- Wide boxes: High client heterogeneity in gradient magnitudes
- Outliers: Clients with unusual gradient patterns (potential data anomalies)

### 5. **Privacy–Utility Trade-off Analysis**

**Description**: Dual-panel analysis showing how differential privacy noise affects both accuracy and fairness.

**Features**:
- Left panel: Noise (σ) vs. Federated Accuracy
- Right panel: Noise (σ) vs. Fairness Disparity
- Multiple noise levels: σ ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3}

**Scientific Value**:
- Quantifies the fundamental privacy-utility trade-off (Dwork & Roth, 2014)
- Reveals optimal noise level for balancing privacy and utility
- Shows how noise affects fairness (may increase or decrease disparity)
- Essential for policy decisions on privacy budgets

**Mathematical Relationship**:

$$
A(\sigma) \approx A(0) \cdot \exp\left(-\frac{\sigma^2}{2\tau^2}\right)
$$

Where $\tau$ is a problem-dependent sensitivity parameter.

**Interpretation**:
- Steep accuracy decline: High sensitivity to noise
- Gradual decline: Robust to noise
- Disparity changes: Noise may affect clients differently
- Optimal σ: Point where privacy gain outweighs utility loss

### 6. **Privacy–Utility–Scalability Frontier (3D Surface/Heatmap)**

**Description**: Three-dimensional visualization showing accuracy as a function of both noise level (σ) and client count.

**Features**:
- 3D interactive surface plot (WebGL) or 2D heatmap fallback
- X-axis: Noise (σ) from 0.0 to 0.3
- Y-axis: Client count {2, 4, 8, 12, 16, 20}
- Z-axis: Federated accuracy
- Color mapping: Blues colormap indicating accuracy levels

**Scientific Value**:
- Reveals three-way trade-offs: privacy, utility, and scalability
- Identifies optimal operating points for different client counts
- Shows how system scales with more participants
- Critical for deployment planning

**Interpretation**:
- Steep surface: High sensitivity to parameters
- Flat regions: Robust operating zones
- Ridges: Optimal parameter combinations
- Valleys: Poor performance regions

### 7. **Risk–Utility Summary (Radar Chart)**

**Description**: Multi-objective comparison between centralized baseline and federated model across normalized dimensions.

**Features**:
- Four normalized axes (0-1 scale):
  - **Accuracy**: $A_{fed}$ (direct)
  - **Fairness**: $1 - \min(D_{fairness}, 1)$ (inverted, higher is better)
  - **Gradient Norm**: $1 - \frac{\|\nabla\|_{mean}}{\|\nabla\|_{max}}$ (inverted, higher is better)
  - **Privacy Noise**: $\frac{\sigma}{0.5}$ (direct, higher is better)
- Baseline (centralized) vs. Current (federated) comparison
- Filled polygons for visual comparison

**Scientific Value**:
- Holistic view of multi-objective trade-offs
- Identifies which objectives are most compromised
- Guides decision-making on acceptable trade-offs
- Useful for stakeholder communication

**Interpretation**:
- Larger baseline area: Centralized model dominates
- Larger current area: Federated model competitive
- Asymmetric shapes: Imbalanced trade-offs
- Overlap: Areas where federated matches centralized

## 🔬 Algorithm Details

### Federated Averaging (FedAvg)

The simulator implements FedAvg, the canonical federated learning algorithm (McMahan et al., 2017).

**Algorithm**:
1. Initialize global model: $w^{(0)} = \mathbf{0}$
2. For each round $t = 1, \ldots, T$:
   - For each client $k = 1, \ldots, K$:
     - Compute local gradient: $\nabla_k^{(t)} = \nabla \mathcal{L}(w^{(t-1)}, \mathcal{D}_k)$
     - Add DP noise: $\tilde{\nabla}_k^{(t)} = \nabla_k^{(t)} + \mathcal{N}(0, \sigma^2 I)$
   - Aggregate gradients: $\bar{\nabla}^{(t)} = \frac{1}{K} \sum_{k=1}^{K} \tilde{\nabla}_k^{(t)}$
   - Update global model: $w^{(t)} = w^{(t-1)} - \eta \bar{\nabla}^{(t)}$

Where:
- $w^{(t)}$: Global model at round $t$
- $\mathcal{D}_k$: Local dataset of client $k$
- $\eta$: Learning rate
- $\sigma$: DP noise standard deviation

### Model: Logistic Regression

Binary classification using logistic regression with sigmoid activation.

**Prediction Function**:

$$
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T \mathbf{x} + b))}
$$

**Loss Function** (Binary Cross-Entropy):

$$
\mathcal{L}(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
$$

**Gradient**:

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})
$$

Where $\hat{\mathbf{y}} = \sigma(\mathbf{X}\mathbf{w} + b)$ is the predicted probability vector.

## 🎯 Example Workflow

1. **Configure Parameters**: Set 8 clients, 1000 samples, 25 rounds, 0.2 learning rate, 0.1 noise
2. **Run Simulation**: Click "Run Simulation" and wait for completion
3. **Review Summary**: Check the risk assessment cards for initial insights
4. **Analyze Plots**: 
   - Examine accuracy convergence with CI bands
   - Review fairness disparity evolution
   - Inspect gradient norm distributions
5. **Advanced Analysis**:
   - Enable "Show Advanced Metrics" for MAE, Gini, and ε
   - Run "Analyze Noise Impact" for privacy-utility trade-off
   - Generate 3D surface for scalability analysis
6. **Download Results**: Save plots for presentations or reports
7. **Experiment**: Try different parameters to explore trade-offs

## 🔧 Technical Details

### Architecture

- **Frontend**: Streamlit web interface with Plotly for interactive visualizations
- **Backend**: Pure Python with NumPy for computations
- **Visualization**: Plotly for interactive charts (supports WebGL for 3D)
- **Data**: Synthetic binary classification data with configurable distribution

### Data Generation

Synthetic data generation for reproducible experiments:

$$
\mathbf{x}_i \sim \mathcal{N}(\mathbf{0}, I), \quad y_i = \mathbb{1}[\sigma(\mathbf{w}^* \mathbf{x}_i + b^*) > 0.5]
$$

Where $\mathbf{w}^*, b^*$ are randomly generated true parameters.

### Risk Metrics Summary

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy Gap** | $\Delta A = A_{central} - A_{fed}$ | Utility cost of federation |
| **Fairness Disparity** | $D = \max_k A_k - \min_k A_k$ | Inter-client bias measure |
| **Gradient Norm** | $\|\nabla_k\|_2$ | Privacy leakage proxy |
| **MAE** | $\frac{1}{T}\sum_t \|A_{central} - A_{fed}^{(t)}\|$ | Persistent deviation |
| **Gini Coefficient** | $G = \frac{n+1-2\sum_i \frac{(n+1-i)A_{(i)}}{\sum_j A_j}}{n}$ | Inequality measure |
| **Privacy Loss** | $\varepsilon \approx \frac{1}{2\sigma^2}$ | DP guarantee strength |

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
   - Disable 3D surface generation for quick exploration

4. **Plot download issues**
   - Ensure kaleido is installed: `pip install kaleido`
   - Check browser download settings
   - Use browser screenshot as fallback

5. **3D surface not rendering**
   - Browser may not support WebGL
   - Use "2D Heatmap (Fallback)" option
   - Check browser console for WebGL errors

### Performance Tips

- Start with smaller datasets for quick exploration
- Use fewer rounds for rapid iteration
- Increase noise levels gradually to see effects
- Cache results by reusing same random seed

## 📚 Scientific References

### Federated Learning

1. **McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017)**. Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS 2017*.  
   [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

2. **Kairouz, P., et al. (2021)**. Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*.  
   [arXiv:1912.04977](https://arxiv.org/abs/1912.04977)

### Differential Privacy

3. **Dwork, C., & Roth, A. (2014)**. The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*.  
   [PDF](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

4. **Abadi, M., et al. (2016)**. Deep Learning with Differential Privacy. *CCS 2016*.  
   [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

### Fairness and Bias

5. **Mitchell, S., et al. (2021)**. Model Cards for Model Reporting. *FAccT 2019*.  
   [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)

6. **Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021)**. A Survey on Bias and Fairness in Machine Learning. *ACM Computing Surveys*.  
   [arXiv:1908.09635](https://arxiv.org/abs/1908.09635)

### Gradient Inversion Attacks

7. **Zhu, L., Liu, Z., & Han, S. (2019)**. Deep Leakage from Gradients. *NeurIPS 2019*.  
   [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)

## 🤝 Contributing

This simulator is designed for the AffectLog 360° Demo and Workshop sessions. For questions or improvements, please contact the development team.

## 📄 License

This project is developed for educational and research purposes as part of the AffectLog initiative.

---

**Built for AffectLog 360° Demo** 🤖 | Explore the trade-offs between privacy, fairness, and utility in decentralized AI systems 

**Part of the Council of Europe – Ministry of Slovenia Hands-On Risk & Rights Lab**
