from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(x):
    C = np.max(x, axis=1, keepdims=True)
    x -= C
    x = np.exp(x)
    s = x / np.sum(x, axis=1, keepdims=True)
    return s

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, C = W.shape
    N = X.shape[0]
    for i in range(N):
        scores = X[i].dot(W).reshape((1, C))
        probs = softmax(scores)
        loss += -np.log(probs[0,y[i]])
        for j in range(C):
            if j == y[i]:
                dW[:, j:j+1] += (-1 + probs[0,j]) * X[i].reshape((D,1))
            else:
                dW[:, j:j+1] += probs[0,j] * X[i].reshape((D,1))

    loss /= N
    dW /= N

    loss += reg * np.sum(W**2)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    D, C = W.shape
    N = X.shape[0]
    
    scores = np.matmul(X, W) 
    probs = softmax(scores)
    
    loss += -np.sum(np.log(probs[range(N), y]))
    loss /= N
    loss += reg * np.sum(W**2)

    weights = probs
    weights[range(N), y] -= 1
    dW += np.matmul(X.T, weights)
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
