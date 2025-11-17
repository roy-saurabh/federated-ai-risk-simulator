#!/usr/bin/env python3
"""
Demo Usage of Federated AI Risk Simulator Functions
==================================================

This script demonstrates how to use the core federated learning functions
programmatically, without the Streamlit interface. Useful for:
- Integration into existing code
- Batch experiments
- Custom analysis workflows

Usage:
    python demo_usage.py
"""

import numpy as np
from federated_ai_risk_simulator import (
    generate_synthetic_data,
    run_federated_simulation,
    create_accuracy_plot,
    create_fairness_plot,
    create_gradient_norms_plot
)


def demo_basic_simulation():
    """Demonstrate basic federated learning simulation."""
    print("🤖 Basic Federated Learning Simulation")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    client_data, test_data = generate_synthetic_data(
        n_clients=5,
        n_samples=1000,
        seed=42
    )
    
    print(f"Created {len(client_data)} clients with {sum(len(X) for X, _ in client_data)} total samples")
    print(f"Test set size: {len(test_data[0])} samples")
    
    # Run simulation
    print("\nRunning federated learning simulation...")
    results = run_federated_simulation(
        client_data=client_data,
        test_data=test_data,
        rounds=20,
        lr=0.2,
        noise=0.1,
        seed=42
    )
    
    # Display results
    print("\n📊 Simulation Results:")
    print(f"Centralized Accuracy: {results['central_accuracy']:.3f}")
    print(f"Federated Accuracy: {results['federated_accuracy']:.3f}")
    print(f"Accuracy Difference: {results['accuracy_difference']:.3f}")
    print(f"Fairness Disparity: {results['final_disparity']:.3f}")
    print(f"Mean Gradient Norm: {results['mean_gradient_norm']:.3f} ± {results['std_gradient_norm']:.3f}")
    
    return results


def demo_noise_impact_analysis():
    """Demonstrate noise impact analysis."""
    print("\n\n🔬 Noise Impact Analysis")
    print("=" * 50)
    
    # Generate data once
    client_data, test_data = generate_synthetic_data(
        n_clients=8,
        n_samples=2000,
        seed=42
    )
    
    # Test different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_by_noise = []
    
    print("Testing different noise levels...")
    for noise in noise_levels:
        print(f"  Testing noise σ = {noise}")
        result = run_federated_simulation(
            client_data=client_data,
            test_data=test_data,
            rounds=25,
            lr=0.2,
            noise=noise,
            seed=42
        )
        result['noise'] = noise
        results_by_noise.append(result)
    
    # Display summary
    print("\n📈 Noise Impact Summary:")
    print("Noise (σ) | Accuracy | Disparity | Gradient Norm")
    print("-" * 50)
    for result in results_by_noise:
        print(f"{result['noise']:8.2f} | {result['federated_accuracy']:8.3f} | {result['final_disparity']:9.3f} | {result['mean_gradient_norm']:12.3f}")
    
    return results_by_noise


def demo_client_analysis():
    """Demonstrate per-client analysis."""
    print("\n\n👥 Per-Client Analysis")
    print("=" * 50)
    
    # Run simulation with more clients
    client_data, test_data = generate_synthetic_data(
        n_clients=10,
        n_samples=1500,
        seed=42
    )
    
    results = run_federated_simulation(
        client_data=client_data,
        test_data=test_data,
        rounds=30,
        lr=0.2,
        noise=0.1,
        seed=42
    )
    
    # Analyze per-client performance
    final_client_accuracies = results['client_accuracies_history'][-1]
    
    print("Final Client Performance:")
    print("Client | Accuracy | Performance")
    print("-" * 35)
    
    for i, acc in enumerate(final_client_accuracies):
        if acc > 0.8:
            performance = "Excellent"
        elif acc > 0.7:
            performance = "Good"
        elif acc > 0.6:
            performance = "Fair"
        else:
            performance = "Poor"
        
        print(f"{i+1:6d} | {acc:8.3f} | {performance}")
    
    # Identify best and worst clients
    best_client = np.argmax(final_client_accuracies)
    worst_client = np.argmin(final_client_accuracies)
    
    print(f"\n🏆 Best performing client: Client {best_client + 1} ({final_client_accuracies[best_client]:.3f})")
    print(f"⚠️  Worst performing client: Client {worst_client + 1} ({final_client_accuracies[worst_client]:.3f})")
    print(f"📊 Fairness disparity: {results['final_disparity']:.3f}")
    
    return results


def demo_hyperparameter_comparison():
    """Demonstrate hyperparameter comparison."""
    print("\n\n⚙️ Hyperparameter Comparison")
    print("=" * 50)
    
    # Generate data
    client_data, test_data = generate_synthetic_data(
        n_clients=6,
        n_samples=1200,
        seed=42
    )
    
    # Test different learning rates
    learning_rates = [0.05, 0.1, 0.2, 0.3, 0.5]
    lr_results = []
    
    print("Testing different learning rates...")
    for lr in learning_rates:
        print(f"  Testing learning rate = {lr}")
        result = run_federated_simulation(
            client_data=client_data,
            test_data=test_data,
            rounds=20,
            lr=lr,
            noise=0.1,
            seed=42
        )
        result['learning_rate'] = lr
        lr_results.append(result)
    
    # Display results
    print("\n📊 Learning Rate Comparison:")
    print("Learning Rate | Accuracy | Disparity | Convergence")
    print("-" * 55)
    
    for result in lr_results:
        # Check if model converged (accuracy in last 5 rounds is stable)
        last_5_acc = result['accuracy_history'][-5:]
        convergence = "Stable" if max(last_5_acc) - min(last_5_acc) < 0.02 else "Unstable"
        
        print(f"{result['learning_rate']:13.2f} | {result['federated_accuracy']:8.3f} | {result['final_disparity']:9.3f} | {convergence}")
    
    return lr_results


def main():
    """Run all demo functions."""
    print("🚀 Federated AI Risk Simulator - Demo Usage")
    print("=" * 60)
    print("This demo shows how to use the core functions programmatically.")
    print()
    
    # Run demos
    basic_results = demo_basic_simulation()
    noise_results = demo_noise_impact_analysis()
    client_results = demo_client_analysis()
    lr_results = demo_hyperparameter_comparison()
    
    print("\n\n✅ All demos completed!")
    print("\n💡 Key Insights:")
    print("- Higher noise levels generally reduce accuracy but improve privacy")
    print("- Learning rate affects both convergence speed and final performance")
    print("- Fairness disparity can vary significantly across different configurations")
    print("- Gradient norms provide insight into information leakage risk")
    
    print("\n🔧 Next Steps:")
    print("- Modify parameters to explore different scenarios")
    print("- Integrate these functions into your own analysis workflows")
    print("- Use the Streamlit interface for interactive exploration")
    print("- Extend the functions for your specific use cases")


if __name__ == "__main__":
    main() 