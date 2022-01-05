"""Single Layer Perceptron
"""

import numpy as np
from optimizer import PerceptronRule
from utils import add_bias, weight_init
from activations import tlu


class SLP:
    """A model for a single layer perceptron.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 5, learning_rate: float = 0.1
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Input samples.
        y : np.ndarray
            Targets for the input samples.
        epochs : int, optional
            Number of training epochs, by default 5
        learning_rate : float, optional
            Value for the learning rate, by default 0.1
        """
        self.X = add_bias(X)
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = weight_init(self.X.shape[-1])
        self.activation = tlu
        self.optimizer = PerceptronRule(self.learning_rate)

    def fit(self):
        """Fit the inputs onto the model and updates the weights.
        """
        print(f"{' Training Process Started ':#^50}")

        for epoch in range(self.epochs):
            print(f"---- Epoch {epoch+1}")

            # initialize error to '0' at the start of each epoch
            error = 0

            # update the weights
            for sample, target in zip(self.X, self.y):
                pred = self.predict(sample)
                target = np.atleast_1d(target)
                self.weights += self.optimizer.delta_w(target, pred, sample)

            # update the error
            for sample, target in zip(self.X, self.y):
                error += self.optimizer.error(target, self.predict(sample))

            print(f"Error = {error}")

        print(f"{' Training Process Ended ':#^50}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the targets for the given sample(s) with the currunt weights.

        Parameters
        ----------
        X : np.ndarray
            Input sample or samples.

        Returns
        -------
        np.ndarray
            The prediction(s) for the given input sample(s).
        """
        # check if bias was added
        if X.shape[-1] == self.weights.shape[0] - 1:
            X = add_bias(X)

        net = np.dot(X, self.weights.T)
        net = np.atleast_1d(net)  # deal with scalars

        return self.activation(net)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Print the predictions and accuracy for the given input sample(s).

        Parameters
        ----------
        X : np.ndarray
            Input sample(s).
        y : np.ndarray
            Desired target(s) for the input sample(s).
        """
        # deal with single sample input:
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # calculate predictions and accuracy
        preds = self.predict(X)
        correct = np.sum(y == preds)
        accuracy = correct / X.shape[0]

        # printing the table
        col = [6, 4 + 2 * X.shape[-1], 8, 4]  # size of each column

        # table header
        print(
            f"{'idx':<{col[0]}}",
            f"{'X':<{col[1]}}",
            f"{'Target':<{col[2]}}",
            f"{'Pred':<{col[3]}}",
            sep="",
        )
        print("=" * sum(col))

        # table values
        for idx, (x, target, pred) in enumerate(zip(X, y, preds)):
            print(
                f"{idx:<{col[0]}}",
                f"{f'{x}':<{col[1]}}",
                f"{target:<{col[2]}}",
                f"{pred:<{col[3]}}",
                sep="",
            )
        print("=" * sum(col), end="\n\n")

        # accuracy
        print(f"Accuracy = {accuracy}")
