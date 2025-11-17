"""
affectlog_risk_demo.py
----------------------

This script extends the federated learning demonstration by incorporating
basic AI risk assessment metrics for decentralised architectures. It
builds upon the original federated training simulation and adds
measurements to help data scientists reason about potential risks
associated with fairness disparities and privacy leakages in a
federated context.

The risk assessment comprises the following aspects:

* **Fairness disparity**: Evaluate the accuracy of the global model on
  each client's local dataset. A large gap between the best and worst
  performing clients suggests the model may be biased toward certain
  data distributions. This simple metric illustrates the need for
  monitoring per‑client outcomes when deploying federated models.

* **Gradient norm statistics**: During training, each client computes
  gradients on local data. The L2 norm of these gradients can hint at
  how much information about the underlying data is being shared. We
  compute the mean and standard deviation of gradient norms across
  clients per round. Larger norms may indicate greater risk of
  information leakage, whereas adding differential privacy noise
  reduces these norms.

* **Accuracy gap**: Compare the accuracy of the global federated
  model with a centrally trained baseline. While not directly a risk
  metric, a significant performance gap could signal underfitting in
  the federated model, which in turn might affect fairness or impact
  user trust.

By integrating these measurements into the simulation, this script
illustrates how one might begin to assess AI risks in a privacy‑
preserving, decentralised architecture. Note that this example uses
synthetic data and simple metrics; real‑world deployments should
incorporate domain‑specific fairness definitions and more rigorous
privacy analyses.

Usage example:
  python3 affectlog_risk_demo.py --clients 5 --samples 800 --rounds 15 --noise 0.05

See the argument parser for details on configurable parameters.
"""

import argparse
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid function."""
    return 1 / (1 + np.exp(-z))


def generate_synthetic_data(n_clients: int, n_samples: int, dim: int = 2, seed: int = 42):
    """
    Generate synthetic binary classification data split across clients.

    Each client receives an approximately equal number of samples. The
    features are drawn from a multivariate normal distribution, and labels
    are assigned based on a linear decision boundary with added noise.

    Parameters
    ----------
    n_clients : int
        Number of federated clients.
    n_samples : int
        Total number of training samples across all clients.
    dim : int, optional
        Number of features per sample (default is 2).
    seed : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    client_data : list of tuples
        Each tuple contains (X_i, y_i) for client i where X_i has shape
        (n_i, dim) and y_i has shape (n_i,).
    test_data : tuple
        A tuple (X_test, y_test) representing held‑out test data.
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
    preds = sigmoid(X @ w)
    error = preds - y
    grad = X.T @ error / len(X)
    return grad


def federated_train_with_metrics(
    client_data,
    rounds: int = 20,
    lr: float = 0.2,
    noise: float = 0.0,
    seed: int = 42,
):
    """
    Train a logistic regression model using federated averaging while
    collecting risk assessment metrics.

    Parameters
    ----------
    client_data : list of tuples
        Local datasets for each client.
    rounds : int
        Number of federated training rounds.
    lr : float
        Learning rate for gradient descent.
    noise : float
        Standard deviation of Gaussian noise added to gradients.
    seed : int
        Random seed for noise generation.

    Returns
    -------
    w_global : np.ndarray
        Trained model weights.
    grad_norms_per_round : list of list of floats
        L2 norms of gradients from each client for each round.
    """
    rng = np.random.default_rng(seed)
    dim = client_data[0][0].shape[1]
    w_global = np.zeros(dim)
    grad_norms_per_round = []
    for _ in range(rounds):
        grads = []
        norms = []
        for X_i, y_i in client_data:
            grad = compute_gradients(w_global, X_i, y_i)
            if noise > 0:
                grad += rng.normal(scale=noise, size=grad.shape)
            norms.append(np.linalg.norm(grad))
            grads.append(grad)
        mean_grad = np.mean(grads, axis=0)
        w_global -= lr * mean_grad
        grad_norms_per_round.append(norms)
    return w_global, grad_norms_per_round


def train_centralised(X: np.ndarray, y: np.ndarray, rounds: int = 20, lr: float = 0.2) -> np.ndarray:
    dim = X.shape[1]
    w = np.zeros(dim)
    for _ in range(rounds):
        grad = compute_gradients(w, X, y)
        w -= lr * grad
    return w


def evaluate_model(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    preds = (sigmoid(X @ w) > 0.5).astype(np.float32)
    return (preds == y).mean()


def assess_fairness_disparity(w: np.ndarray, client_data) -> float:
    """
    Compute the range of accuracies across clients using the global model.

    A large disparity suggests the model may favour some data
    distributions over others.
    """
    accuracies = []
    for X_i, y_i in client_data:
        acc_i = evaluate_model(w, X_i, y_i)
        accuracies.append(acc_i)
    return max(accuracies) - min(accuracies)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated learning demo with AI risk assessment for AffectLog presentation"
    )
    parser.add_argument("--clients", type=int, default=3, help="Number of federated clients")
    parser.add_argument("--samples", type=int, default=600, help="Total number of training samples")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise for differential privacy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    # Generate synthetic data
    client_data, (X_test, y_test) = generate_synthetic_data(
        n_clients=args.clients, n_samples=args.samples, seed=args.seed
    )
    # Centralised model
    X_full = np.vstack([X for X, _ in client_data])
    y_full = np.hstack([y for _, y in client_data])
    w_central = train_centralised(X_full, y_full, rounds=args.rounds, lr=args.lr)
    acc_central = evaluate_model(w_central, X_test, y_test)
    print(f"Centralised training accuracy: {acc_central:.4f}")
    # Federated training with metrics
    w_global, grad_norms = federated_train_with_metrics(
        client_data, rounds=args.rounds, lr=args.lr, noise=args.noise, seed=args.seed
    )
    acc_fed = evaluate_model(w_global, X_test, y_test)
    print(f"Federated training accuracy: {acc_fed:.4f}")
    # Risk assessment
    disparity = assess_fairness_disparity(w_global, client_data)
    print(f"Fairness disparity across clients: {disparity:.4f}")
    # Compute gradient norm statistics
    all_norms = np.array(grad_norms)
    mean_norm = all_norms.mean()
    std_norm = all_norms.std()
    print(f"Average gradient L2 norm: {mean_norm:.4f} (std: {std_norm:.4f})")
    diff = acc_central - acc_fed
    print(f"Accuracy difference (centralised - federated): {diff:.4f}")
    print(f"Model weights (federated): {w_global}")


if __name__ == "__main__":
    main()