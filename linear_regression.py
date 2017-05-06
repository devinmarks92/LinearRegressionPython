import numpy as np
import matplotlib.pyplot as plt

def normal_equation(X, y):
    """
    Find the optimum theta without iteration.
    The cost can be minimized by explicitly taking its derivatives with respect
    to theta values and setting them to zero.
    """
    try:
        theta = np.linalg.pinv(X.T * X) * X.T * y
        return theta
    except np.linalg.LinAlgError:
        print 'SVD function in normal equation did not converge.'
    return np.zeros((m, 1))

def normalize(X, parameters, m):
    """
    Normalize the X values and the prediction parameters.
    Speed up gradient descent by having each of the input values roughly in the
    same range.
    """
    mu = np.mean(X[:,1:m], axis=0)
    sigma = np.std(X[:,1:m], axis=0)
    X[:,1:m] = np.divide((X[:,1:m] - mu), sigma)
    parameters[1:m] = np.divide((parameters[1:m] - mu), sigma)
    return X, parameters

def cost(X, y, m, theta):
    """
    Cost of the current hypothesis.
    Take an average difference of all the results of the hypothesis with inputs
    from X the actual outputs y.
    """
    hypothesis = X * theta
    J = sum(np.power((hypothesis - y), 2) / (2 * m))
    return J

def gradient_descent(X, y, m, alpha, num_iter):
    """
    Take the derivative and find the tangential line of the cost function to
    take a step (alpha) down the cost function in the direction with the
    steepest descent.
    """
    J_history = []
    theta = np.zeros((m, 1))
    for _ in range(num_iter):
        theta = theta - alpha * (X.T * ((X * theta) - y) / m)
        J_history.append(float(cost(X, y, m, theta)))
    return theta, J_history

def plot_descent(iters, J_history):
    """Plot cost as gradient descent iterates."""
    plt.plot(iters, J_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
