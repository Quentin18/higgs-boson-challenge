"""Loss function."""


def compute_loss(y, tx, w):
    """Calculate the loss."""
    n = len(y)
    e = y - (tx @ w)
    return (e @ e) / (2 * n)
