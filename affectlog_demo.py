"""
affectlog_demo.py
------------------

This script implements a minimal federated learning simulation using pure
NumPy. It showcases how multiple clients can collaboratively train a
logistic regression model without sharing raw data. Each client holds
its own subset of synthetic data. During each round of federated
training, clients compute gradients on their local data, optionally add
differential privacy (DP) noise, and send these gradients to a central
server. The server aggregates the gradients by averaging them and
updates a global model. After a number of rounds, the global model is
evaluated on a held‑out test set.

This demonstration aligns with the AffectLog presentation by
illustrating how privacy‑preserving, decentralized learning can be
implemented in practice. You can experiment with the number of
clients, number of federated rounds, and the amount of DP noise to see
how these factors affect model performance.

Usage:
  python3 affectlog_demo.py --clients 5 --rounds 10 --noise 0.1

Arguments:
  --clients N      Number of federated clients (default: 3)
  --samples N      Total number of training samples across all clients (default: 600)
  --rounds N       Number of federated training rounds (default: 20)
  --lr FLOAT       Learning rate for model updates (default: 0.2)
  --noise FLOAT    Standard deviation of Gaussian noise added to gradients for
                   differential privacy (default: 0.0, meaning no noise)
  --seed INT       Random seed for reproducibility (default: 42)

Example:
  python3 affectlog_demo.py --clients 4 --samples 800 --rounds 15 --noise 0.05

The script prints the accuracy of the global model on the test set
after training. It also prints the centralised baseline accuracy
obtained by training on the entire dataset without federation for
comparison.
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
    # Create a random linear model for generating labels
    true_w = rng.normal(size=(dim,))
    true_b = rng.normal()
    # Generate features
    X = rng.normal(size=(n_samples, dim))
    # Compute probabilities and labels
    probs = sigmoid(X @ true_w + true_b)
    y = (probs > 0.5).astype(np.float32)
    # Shuffle data
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    # Split into training and test sets (80/20 split)
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # Partition training data among clients
    client_data = []
    sizes = np.full(n_clients, len(X_train) // n_clients, dtype=int)
    sizes[: len(X_train) % n_clients] += 1
    start = 0
    for sz in sizes:
        X_i = X_train[start : start + sz]
        y_i = y_train[start : start + sz]
        client_data.append((X_i, y_i))
        start += sz
    test_data = (X_test, y_test)
    return client_data, test_data


def compute_gradients(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the logistic loss for a single batch.

    Parameters
    ----------
    w : np.ndarray
        Current model weights of shape (dim,).
    X : np.ndarray
        Features of shape (n_samples, dim).
    y : np.ndarray
        Binary labels of shape (n_samples,).

    Returns
    -------
    grad : np.ndarray
        Gradient of the loss with respect to the weights.
    """
    preds = sigmoid(X @ w)
    error = preds - y
    grad = X.T @ error / len(X)
    return grad


def federated_train(
    client_data,
    rounds: int = 20,
    lr: float = 0.2,
    noise: float = 0.0,
    seed: int = 42,
):
    """
    Train a logistic regression model using federated averaging.

    Parameters
    ----------
    client_data : list of tuples
        List of (X_i, y_i) pairs for each client.
    rounds : int
        Number of federated training rounds.
    lr : float
        Learning rate for gradient descent updates.
    noise : float
        Standard deviation of Gaussian noise to add to gradients for
        differential privacy (default is 0.0, meaning no noise).
    seed : int
        Random seed for noise generation.

    Returns
    -------
    w_global : np.ndarray
        Trained global model weights.
    """
    rng = np.random.default_rng(seed)
    dim = client_data[0][0].shape[1]
    w_global = np.zeros(dim)
    for rnd in range(rounds):
        grads = []
        for X_i, y_i in client_data:
            grad = compute_gradients(w_global, X_i, y_i)
            if noise > 0:
                grad += rng.normal(scale=noise, size=grad.shape)
            grads.append(grad)
        # Average gradients and update global model
        mean_grad = np.mean(grads, axis=0)
        w_global -= lr * mean_grad
    return w_global


def train_centralised(X: np.ndarray, y: np.ndarray, rounds: int = 20, lr: float = 0.2) -> np.ndarray:
    """
    Train a logistic regression model on the entire dataset centrally.

    Parameters
    ----------
    X : np.ndarray
        Features of shape (n_samples, dim).
    y : np.ndarray
        Binary labels.
    rounds : int
        Number of gradient descent steps.
    lr : float
        Learning rate.

    Returns
    -------
    w : np.ndarray
        Trained model weights.
    """
    dim = X.shape[1]
    w = np.zeros(dim)
    for _ in range(rounds):
        grad = compute_gradients(w, X, y)
        w -= lr * grad
    return w


def evaluate_model(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate a logistic regression model and return accuracy.

    Parameters
    ----------
    w : np.ndarray
        Model weights.
    X : np.ndarray
        Test features.
    y : np.ndarray
        True labels.

    Returns
    -------
    acc : float
        Classification accuracy (between 0 and 1).
    """
    preds = (sigmoid(X @ w) > 0.5).astype(np.float32)
    acc = (preds == y).mean()
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Federated learning demo for AffectLog presentation")
    parser.add_argument("--clients", type=int, default=3, help="Number of federated clients")
    parser.add_argument("--samples", type=int, default=600, help="Total number of training samples")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated training rounds")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--noise", type=float, default=0.0, help="Std dev of Gaussian noise for DP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    # Generate data
    client_data, (X_test, y_test) = generate_synthetic_data(
        n_clients=args.clients, n_samples=args.samples, seed=args.seed
    )
    # Centralised baseline
    X_train_full = np.vstack([X for X, _ in client_data])
    y_train_full = np.hstack([y for _, y in client_data])
    w_central = train_centralised(X_train_full, y_train_full, rounds=args.rounds, lr=args.lr)
    acc_central = evaluate_model(w_central, X_test, y_test)
    print(f"Centralised training accuracy: {acc_central:.4f}")
    # Federated training
    w_global = federated_train(
        client_data, rounds=args.rounds, lr=args.lr, noise=args.noise, seed=args.seed
    )
    acc_fed = evaluate_model(w_global, X_test, y_test)
    print(f"Federated training accuracy: {acc_fed:.4f}")
    # Report difference
    diff = acc_central - acc_fed
    print(f"Accuracy difference (centralised - federated): {diff:.4f}")
    print(f"Model weights (federated): {w_global}")


if __name__ == "__main__":
    main()