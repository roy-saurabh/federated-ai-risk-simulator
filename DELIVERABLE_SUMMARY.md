# Federated AI Risk Simulator - Complete Deliverable

## 🎯 Overview

This deliverable provides a comprehensive, interactive web application for simulating federated learning scenarios with AI risk assessment metrics. The tool allows data scientists to explore the trade-offs between privacy, fairness, and utility in decentralized AI systems.

## 📦 Complete Package Contents

### Core Application Files
- **`federated_ai_risk_simulator.py`** - Main Streamlit application
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Comprehensive documentation

### Launcher Scripts
- **`run_simulator.py`** - Python launcher with dependency checking
- **`run_simulator.bat`** - Windows batch file launcher
- **`run_simulator.sh`** - Unix/Linux/macOS shell script launcher

### Demo and Examples
- **`demo_usage.py`** - Programmatic usage examples
- **`affectlog_demo.py`** - Original federated learning demo
- **`affectlog_risk_demo.py`** - Original risk assessment demo

## 🚀 Quick Start Instructions

### Option 1: Simple Launch (Recommended)
```bash
# On macOS/Linux:
./run_simulator.sh

# On Windows:
run_simulator.bat

# Or using Python launcher:
python run_simulator.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run federated_ai_risk_simulator.py
```

### Option 3: Programmatic Usage
```bash
# Run demo examples
python demo_usage.py
```

## 🎛️ Key Features

### 1. Interactive Parameter Configuration
- **Number of Clients**: 2-20 federated learning participants
- **Total Samples**: 100-10,000 training data points
- **Federated Rounds**: 1-50 training iterations
- **Learning Rate**: 0.01-1.0 gradient descent step size
- **DP Noise (σ)**: 0.0-0.5 differential privacy noise level
- **Random Seed**: For reproducible results

### 2. Comprehensive Risk Assessment
- **Centralized vs Federated Accuracy**: Performance comparison
- **Fairness Disparity**: Bias measurement across clients
- **Gradient Norms**: Information leakage risk assessment
- **Privacy-Utility Trade-off**: Impact of differential privacy

### 3. Real-time Visualizations
- **Accuracy Over Rounds**: Model performance progression
- **Fairness Disparity Over Rounds**: Bias evolution tracking
- **Gradient Norms Distribution**: Privacy risk visualization
- **Per-Client Analysis**: Individual client performance

### 4. Download Capabilities
- Export all plots as PNG files
- Ready for PowerPoint presentations
- High-quality graphics for reports

## 📊 Risk Metrics Explained

### Fairness Disparity
- **Definition**: max(client_accuracies) - min(client_accuracies)
- **Interpretation**: Higher values indicate more bias across clients
- **Risk Levels**:
  - 🟢 Low: < 0.1 (fair performance across clients)
  - 🟡 Medium: 0.1-0.2 (moderate bias)
  - 🔴 High: > 0.2 (significant bias)

### Gradient Norms
- **Definition**: L2 norm of client gradients
- **Interpretation**: Higher values suggest greater information leakage risk
- **Risk Levels**:
  - 🟢 Low: < 0.2 (good privacy protection)
  - 🟡 Medium: 0.2-0.5 (moderate risk)
  - 🔴 High: > 0.5 (high information leakage risk)

### Accuracy Gap
- **Definition**: centralized_accuracy - federated_accuracy
- **Interpretation**: Performance impact of federated learning
- **Risk Levels**:
  - 🟢 Low: |gap| < 0.05 (minimal performance impact)
  - 🟡 Medium: 0.05 ≤ |gap| < 0.1 (moderate impact)
  - 🔴 High: |gap| ≥ 0.1 (significant performance degradation)

## 🎯 Example User Workflow

### Step 1: Initial Configuration
1. Set parameters: 8 clients, 1000 samples, 25 rounds, 0.2 learning rate, 0.1 noise
2. Click "Run Simulation"
3. Wait for completion (typically 5-10 seconds)

### Step 2: Review Risk Assessment
1. Check the four metric cards for initial insights
2. Review risk analysis explanations
3. Note any high-risk indicators

### Step 3: Analyze Visualizations
1. Examine accuracy progression over rounds
2. Check fairness disparity trends
3. Review gradient norm distributions
4. Download plots for presentations

### Step 4: Experiment and Compare
1. Try different noise levels to see privacy-utility trade-offs
2. Adjust learning rates to optimize convergence
3. Compare results across different configurations

## 🔬 Advanced Features

### Privacy-Utility Trade-off Analysis
- Optional analysis showing noise impact on performance
- Demonstrates fundamental trade-off between privacy and utility
- Useful for policy decisions and parameter tuning

### Per-Client Analysis
- Individual client performance tracking
- Bias detection across client populations
- Fairness monitoring throughout training

### Hyperparameter Optimization
- Learning rate impact on convergence
- Noise level effects on privacy and accuracy
- Client count influence on fairness

## 📈 Use Cases

### For Researchers
- Explore federated learning dynamics
- Study privacy-utility trade-offs
- Validate fairness metrics
- Compare different federation strategies

### For Practitioners
- Assess deployment risks
- Optimize hyperparameters
- Communicate risks to stakeholders
- Design privacy-preserving systems

### For Educators
- Demonstrate federated learning concepts
- Illustrate AI risk assessment
- Show privacy-preserving techniques
- Visualize complex trade-offs

## 🛠️ Technical Architecture

### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive visualizations
- **Custom CSS**: Professional styling

### Backend
- **NumPy**: Efficient numerical computations
- **Pandas**: Data manipulation and analysis
- **Pure Python**: No external ML frameworks required

### Algorithm
- **Model**: Logistic regression with gradient descent
- **Federation**: FedAvg (Federated Averaging)
- **Privacy**: Gaussian differential privacy
- **Evaluation**: Accuracy on held-out test set

## 🚨 Troubleshooting

### Common Issues
1. **Port already in use**: Use `--server.port 8502` flag
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Slow performance**: Reduce samples or rounds
4. **Plot download issues**: Ensure kaleido is installed

### Performance Tips
- Start with smaller datasets for quick exploration
- Use fewer rounds for rapid iteration
- Increase noise levels gradually to see effects

## 📚 Integration with AffectLog

### Presentation Ready
- All visualizations can be downloaded as PNG files
- Professional styling suitable for presentations
- Clear risk assessment metrics for stakeholders

### Workshop Friendly
- Interactive interface for hands-on learning
- Real-time parameter adjustment
- Immediate visual feedback

### Research Compatible
- Programmatic access to core functions
- Extensible architecture for custom analysis
- Reproducible results with seed control

## 🎉 Success Metrics

The simulator successfully demonstrates:
- ✅ Interactive federated learning simulation
- ✅ Comprehensive risk assessment metrics
- ✅ Real-time visualization capabilities
- ✅ Download functionality for presentations
- ✅ Privacy-utility trade-off analysis
- ✅ Professional UI/UX design
- ✅ Cross-platform compatibility
- ✅ Educational value for workshops

## 🔮 Future Enhancements

### Potential Extensions
- Custom dataset upload functionality
- Advanced privacy mechanisms (secure aggregation)
- More sophisticated fairness metrics
- Multi-objective optimization
- Real-time collaboration features

### Integration Opportunities
- Connect with existing ML pipelines
- API endpoints for programmatic access
- Database integration for result storage
- Cloud deployment options

---

## 📞 Support and Contact

This simulator is developed for the AffectLog 360° Demo and Workshop sessions. For questions, improvements, or integration support, please contact the development team.

**Built for AffectLog 360° Demo** 🤖 | Explore the trade-offs between privacy, fairness, and utility in decentralized AI systems 