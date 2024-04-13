import numpy as np
import itertools

###########################################################
#GENERAL UTILS
###########################################################
def one_hot_encode(x):
    """
    One-hot encodes an array of labels.

    Parameters:
    - x (array-like): The input array of labels.

    Returns:
    - X (ndarray): The one-hot encoded array.

    Example:
    >>> labels = [0, 1, 2, 1, 0]
    >>> one_hot_encode(labels)
    array([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])
    """
    # Map the labels to integers starting from 0
    unique_labels, x_int = np.unique(x, return_inverse=True)
    dim = len(unique_labels)
    X = np.zeros((len(x_int), dim))
    X[np.arange(len(x_int)), x_int] = 1
    X = X.astype(int)
    return X

def loss(true_labels, predicted_labels):
    """
    Calculate the label-permutation invariant loss between true labels and predicted labels.

    Parameters:
    true_labels (array-like): The true labels of the nodes.
    predicted_labels (array-like): The predicted labels of the nodes.

    Returns:
    float: The normalized loss value.
    tuple: The best permutation of labels that minimizes the loss.

    The function calculates the loss by finding the permutation of labels
    that minimizes the misclassification rate between the true labels and the permuted predicted labels.
    The normalized loss value is the minimum L1 norm divided by the number of nodes.
    The best permutation is returned as a tuple.

    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    n = len(true_labels)  # Number of nodes
    k = len(np.unique(true_labels))  # Number of communities

    #check compatibility of true and predicted labels
    assert n == len(predicted_labels), "The number of true labels and predicted labels must be the same."
    assert k == len(np.unique(predicted_labels)), "The number of true labels and predicted labels must be the same."

    min_norm = np.inf  # Initialize min_norm to infinity
    best_permutation = None  # Initialize best_permutation to None
    # Loop over all permutations of labels
    for permutation in itertools.permutations(range(k)):
        permuted_labels = [permutation[label] for label in predicted_labels]
        norm = np.sum(true_labels != permuted_labels)
        if norm < min_norm:
            min_norm = norm
            best_permutation = permutation

    return min_norm/n, best_permutation

def Renyi_divergence(p, q, alpha=0.5):
    """
    Calculate the Renyi divergence between two probability distributions.

    Parameters:
    p (array-like): The first probability distribution.
    q (array-like): The second probability distribution.
    alpha (float, optional): The order of the Renyi divergence. Default is 0.5.

    Returns:
    float: The Renyi divergence between p and q.

    """
    return 1/(alpha-1) * np.log(np.sum(p**alpha * q**(1-alpha)))

def Bernoulli_Renyi_divergence(p, q, alpha=0.5):
    """
    Calculate the Bernoulli-Renyi divergence between two Bernoulli distributions.

    Parameters:
    p (float): The probability parameter of the first Bernoulli distribution.
    q (float): The probability parameter of the second Bernoulli distribution.
    alpha (float, optional): The order of the Renyi divergence. Default is 0.5.

    Returns:
    float: The Bernoulli-Renyi divergence between the two distributions.
    """
    return Renyi_divergence(np.array([p, 1-p]), np.array([q, 1-q]), alpha)

###########################################################
#SBM UTILS
###########################################################

def generate_sbm(n, k, P, pi=None, z_t=None, return_Z=False):
    """
    Generate a Stochastic Block Model (SBM) graph.

    Parameters:
    - n (int): Number of nodes in the graph.
    - k (int): Number of communities in the graph.
    - P (numpy.ndarray): Probability matrix of shape (k, k) representing the edge probabilities between communities.
    - pi (numpy.ndarray, optional): probability vector of shape (k,) of assignment probabilities.
    - z_t (numpy.ndarray, optional): Initial community assignments of shape (n,) for each node.
    - return_Z (bool, optional): Whether to return the community assignments also as a one-hot encoded matrix.

    Returns:
    - A (numpy.ndarray): Adjacency matrix of shape (n, n) representing the generated graph.
    - z (numpy.ndarray): Community assignments of shape (n,) for each node.
    - Z (numpy.ndarray, optional): One-hot encoded matrix of shape (n, k) representing the community assignments.

    If `pi` is not provided, a uniform prior is assumed.
    If `z_t` is not provided, random community assignments are generated.
    If `return_Z` is set to True, the function also returns the one-hot encoded matrix `Z`.
    """
    # Assign nodes to communities
    if z_t is None:
        z = assignments(n, k, pi)
    else:
        z = z_t
    # Generate adjacency matrix
    A = generate_adjacency_matrix(z, P)
    if return_Z:
        return A, z, one_hot_encode(z)
    else:
        return A, z

def assignments(n, k, pi=None):
    """
    Generate random assignments of nodes to communities.

    Parameters:
    - n (int): Number of nodes.
    - k (int): Number of communities.
    - pi (ndarray, optional): assignment probabilities. If not provided, a uniform distribution is used.

    Returns:
    - z (ndarray): Array of length n containing the community assignments for each node.
    """
    if pi is None:
        pi = np.ones(k) / k
    z = np.random.choice(a=list(range(k)), size=n, p=pi)
    return z

def generate_adjacency_matrix(z, P):
    """
    Generate an adjacency matrix based on the given community assignments and edge probabilities.

    Parameters:
    z (list): List of community assignments for each node.
    P (numpy.ndarray): Probability matrix representing the edge probabilities between communities.

    Returns:
    numpy.ndarray: Adjacency matrix representing the generated graph.
    """
    n = len(z)
    # Initialize the adjacency matrix
    A = np.zeros((n, n), dtype=int)
    # Generate the adjacency matrix
    for i in range(n):
        for j in range(i+1, n):
            # Generate an edge with probability P[z[i], z[j]]
            if np.random.rand() <= P[z[i], z[j]]:
                A[i, j] = 1
                A[j, i] = 1  # The adjacency matrix is symmetric
    return A

def generate_homogeneous_sbm(n, k, p, q, pi=None, z_t=None):
    """
    Generate a Homogeneous Stochastic Block Model (SBM) graph.

    Parameters:
    - n (int): Number of nodes in the graph.
    - k (int): Number of communities in the graph.
    - p (float): Probability of an edge between nodes in the same community.
    - q (float): Probability of an edge between nodes in different communities.
    - pi (numpy.ndarray, optional): probability vector of shape (k,) of assignment probabilities.
    - z_t (numpy.ndarray, optional): Initial community assignments of shape (n,) for each node.

    Returns:
    - A (numpy.ndarray): Adjacency matrix of shape (n, n) representing the generated graph.
    - z (numpy.ndarray): Community assignments of shape (n,) for each node.

    If `pi` is not provided, a uniform prior is assumed.
    If `z_t` is not provided, random community assignments are generated.
    """
    if q > p:
        raise ValueError("SBM is not assortative.")
    P = np.ones((k, k)) * q + np.eye(k) * (p - q)
    return generate_sbm(n, k, P, pi, z_t)

def random_warm_initializer(z_true, alpha, n, k):
    """
    Generate a random warm initialization of community assignments with a given misclassification rate.
    alpha is a parameter between 0 and 1, and it represents the (average) initial misclassification rate of the generated assignments.
    
    Parameters:
    - z_true (ndarray): True community assignments.
    - alpha (float): Misclassification rate.
    - n (int): Number of nodes.
    - k (int): Number of communities.

    Returns:
    - z (ndarray): Random warm initialization of community assignments.
    """
    z = np.zeros(n)
    for i in range(n):
        if np.random.rand() <= 1 - alpha:
            z[i] = z_true[i]
        else:
            z[i] = np.random.choice([j for j in range(k) if j != z_true[i]])
    return z.astype(int)

def warm_initializer(z_true, alpha, n, k):
    """
    Generate a warm initialization of community assignments with a given misclassification rate.
    alpha is a parameter between 0 and 1, and it represents the initial misclassification rate of the generated assignments.
    
    Parameters:
    - z_true (ndarray): True community assignments.
    - alpha (float): Misclassification rate.
    - n (int): Number of nodes.
    - k (int): Number of communities.

    Returns:
    - z (ndarray): Warm initialization of community assignments.
    """
    z = np.zeros(n)
    # exactly alpha*n misclassified nodes are chosen (floored value)
    misclassified = np.random.choice(n, size=int(alpha*n), replace=False)
    for i in range(n):
        if i in misclassified:
            z[i] = np.random.choice([j for j in range(k) if j != z_true[i]])
        else:
            z[i] = z_true[i]
    return z.astype(int)


