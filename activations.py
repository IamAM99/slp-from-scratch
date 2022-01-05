"""Activation functions for the SLP.
"""

import numpy as np


def tlu(net: np.ndarray) -> np.ndarray:
    """TLU activation function.

    Parameters
    ----------
    net : np.ndarray
        Values of 'net' as a numpy array.

    Returns
    -------
    np.ndarray
        The outputs of activation function for the given 'net' array.
    """
    return np.where(net >= 0, 1, 0)
