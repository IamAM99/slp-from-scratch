"""Necessary functions for SLP.
"""

import numpy as np


def add_bias(matrix: np.ndarray) -> np.ndarray:
    """Add a '1' column to the start of a matrix as bias.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Output matrix with added bias.
    """
    # if 'matrix' is 1D-array, convert it to 2D-array:
    matrix = np.atleast_2d(matrix)
    return np.c_[np.ones(matrix.shape[0]), matrix]


def weight_init(size: int) -> np.ndarray:
    """Initialize the weights array with random numbers.
    
    Parameters
    ----------
    size : int
        The desired size of the array.

    Returns
    -------
    np.ndarray
        An array of random numbers with the specified size.
    """
    return np.random.rand(size)


def add_noise(pattern: np.ndarray, alpha: float) -> np.ndarray:
    """Add noise to a pattern with 'alpha' as the percentage of values in the pattern to change.


    Parameters
    ----------
    pattern : np.ndarray
        Input pattern to add noise to. The input pattern should only have values of 0 and 1.
    alpha : float
        Percentage of values to flip in the pattern.

    Returns
    -------
    np.ndarray
        Noisy pattern.
    """
    noisy_pattern = np.copy(pattern)
    idx = np.random.choice(
        range(noisy_pattern.shape[0]), size=int(alpha * noisy_pattern.shape[0])
    )

    for i in idx:
        if noisy_pattern[i] == 1:
            noisy_pattern[i] = 0
        else:
            noisy_pattern[i] = 1

    return noisy_pattern
