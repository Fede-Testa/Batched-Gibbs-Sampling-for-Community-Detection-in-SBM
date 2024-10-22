from utils import *
import numpy as np

# possible improvement for code consistency: treat the sbm implementation as a class instead of a set of functions

def generate_homogeneous_sbm(n, k, p, q, pi=None, z_t=None, Neal=False, gamma=None):
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
    return generate_sbm(n, k, P, pi=pi, z_t=z_t, Neal=Neal, gamma=gamma)


def generate_sbm(n, k, P, pi=None, z_t=None, return_Z=False, Neal=False, gamma=None):
    """
    Generate a Stochastic Block Model (SBM) graph, given the connectivity probabilities and the number of communities.

    Parameters:
    - n (int): Number of nodes in the graph.
    - k (int): Number of communities in the graph.
    - P (numpy.ndarray): Probability matrix of shape (k, k) representing the edge probabilities between communities.
    - pi (numpy.ndarray, optional): probability vector of shape (k,) of assignment probabilities.
    - z_t (numpy.ndarray, optional): Given community assignments of shape (n,) for each node.
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
        if Neal:
            z = assignments_neal(n, k, gamma)
        else:
            z = assignments(n, k, pi)
    else:
        z = z_t
    # Generate adjacency matrix
    A = generate_adjacency_matrix(z, P)
    if return_Z:
        return A, z, one_hot_encode(z)
    else:
        return A, z


def assignments_neal(n, k, gamma=None):
    """
    Generate random assignments of nodes to communities using the Neal's algorithm.

    Parameters:
    - n (int): Number of nodes.
    - k (int): Number of communities.
    - gamma (float, optional): The parameter of the Dirichlet distribution. If not provided, a uniform distribution is used.

    Returns:
    - z (ndarray): Array of length n containing the community assignments for each node.
    """
    if gamma is None:
        gamma = np.ones(k)
    z = np.zeros(n)
    p = np.random.dirichlet(alpha=gamma)
    for i in range(n):
        z[i] = np.random.choice(a=list(range(k)), p=p)
    return z.astype(int)


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
    # Generate the adjacency matrix (indexing ensures 0 diagonal elements)
    for i in range(n):
        for j in range(i+1, n):
            # Generate an edge with probability P[z[i], z[j]]
            if np.random.rand() <= P[z[i], z[j]]:
                A[i, j] = 1
                A[j, i] = 1  # The adjacency matrix is symmetric
    return A
