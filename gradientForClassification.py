import numpy as np


def compute_gradient(X, y, w, b, *argv):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ###
    f = np.array([])

    for i in range(X.shape[0]):
        result = sigmoid(np.dot(X[i, :], w) + b)
        f = np.append(f, result)

    dj_db = (1 / m) * np.sum(f - y)

    for j in range(n):
        # You code here to calculate the gradient from the i-th example for j-th attribute
        dj_dw[j] = np.sum((f - y) * X[:, j]) / m

    ### END CODE HERE ###

    return dj_db, dj_dw