"""Gradient functions."""


def compute_least_squares_gradient(y, tx, w):
    """Compute the least squares gradient."""
    n = len(y)
    e = y - (tx @ w)
    return -(tx.T @ e) / n
