import numpy as np


class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out = 1 / (1 + np.exp(-1 * x))
        self.cache = out
        cache = out
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = cache * (1 - cache)
        return dout * dx


class Relu:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = np.maximum(np.zeros(x.shape), x)
        self.cache = out
        return out, out

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        diff_relu = np.vectorize(lambda x: 0 if x == 0 else 1)
        return dout * diff_relu(cache)


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    x_re = x.reshape(x.shape[0], -1)
    out = x_re @ w + b
    return out, (x, w, b)


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    # Wx + b
    x, w, b = cache

    N = x.shape[0]
    D = np.prod(x.shape[1:])

    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape(N, D).T @ dout
    db = dout.T @ np.ones(N)

    return dx, dw, db