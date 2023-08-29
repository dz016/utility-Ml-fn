import numpy as np


def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost
    """

    m, n = X.shape

    ### START CODE HERE ###

    f = np.array([])

    for i in range(X.shape[0]):
        # Calculate the result of the sigmoid function and append it to 'f'
        result = sigmoid(np.dot(X[i, :], w) + b)
        f = np.append(f, result)

    loss = (-y * np.log(f)) - ((1 - y) * np.log(1 - f))

    total_cost = (1 / m) * np.sum(loss)

    ### END CODE HERE ###

    return total_cost