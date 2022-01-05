"""Optimizers for updating network weights
"""

import numpy as np


class PerceptronRule:
    """The perceptron rule for updating network weights.
    This rule is explained in [Zurada, Jacek M. Introduction to Artificial Neural Systems, page 64]
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def delta_w(
        self, target: np.ndarray, pred: np.ndarray, sample: np.ndarray
    ) -> np.ndarray:
        """Calculates the delta_weights for updating the weights.

        Parameters
        ----------
        target : np.ndarray
            Target value of the sample.
        pred : np.ndarray
            Predicted value of the sample.
        sample : np.ndarray
            The input sample.

        Returns
        -------
        np.ndarray
            The updating matrix using the perceptron rule.
        """
        return self.learning_rate * (target - pred) * sample

    @staticmethod
    def error(target: np.ndarray, pred: np.ndarray) -> int:
        """The error for the perceptron learning rule. This error is binary, and
        returns '1' when target != pred.


        Parameters
        ----------
        target : np.ndarray
            Target value of the sample.
        pred : np.ndarray
            Predicted value of the sample.

        Returns
        -------
        int
            '1' if target != pred, '0' otherwise.
        """
        if target != pred:
            return 1
        return 0
