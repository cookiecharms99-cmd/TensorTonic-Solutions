import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for step in range(steps):
        z = X @ w.T + b
        p = _sigmoid(z)
        Dz_L = -1/N * (y - p)
        w -= lr * X.T @ Dz_L
        b -= lr * N * np.mean(Dz_L) 
    return w,b 