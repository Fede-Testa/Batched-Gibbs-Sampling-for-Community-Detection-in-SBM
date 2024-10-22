from utils import *
import numpy as np
import itertools
import networkx as nx
from networkx.algorithms.community.quality import modularity
from collections import Counter
from math import log2

# def loss(true_labels, predicted_labels):
#     """
#     Calculate the label-permutation invariant loss between true labels and predicted labels.

#     Parameters:
#     true_labels (array-like): The true labels of the nodes.
#     predicted_labels (array-like): The predicted labels of the nodes.

#     Returns:
#     float: The normalized loss value.
#     tuple: The best permutation of labels that minimizes the loss.

#     The function calculates the loss by finding the permutation of labels
#     that minimizes the misclassification rate between the true labels and the permuted predicted labels.
#     The normalized loss value is the minimum L1 norm divided by the number of nodes.
#     The best permutation is returned as a tuple.

#     """
#     true_labels = np.array(true_labels)
#     predicted_labels = np.array(predicted_labels)
#     n = len(true_labels)  # Number of nodes
#     k = len(np.unique(true_labels))  # Number of communities

#     # check compatibility of true and predicted labels
#     assert n == len(predicted_labels), "The number of true labels and predicted labels must be the same."
#     # assert k == len(np.unique(predicted_labels)), "The number of communities of true labels and predicted labels must be the same."

#     min_norm = np.inf  # Initialize min_norm to infinity
#     best_permutation = None  # Initialize best_permutation to None
#     # Loop over all permutations of labels
#     for permutation in itertools.permutations(range(k)):
#         permuted_labels = [permutation[label] for label in predicted_labels]
#         norm = np.sum(true_labels != permuted_labels)
#         if norm < min_norm:
#             min_norm = norm
#             best_permutation = permutation

#     return min_norm/n, best_permutation


def loss(true_labels, predicted_labels, verbose = False):
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
    # k = len(np.unique(true_labels))  # Number of communities
    # k_pred = len(np.unique(predicted_labels))  # Number of predicted communities
    k_pred = max(max(true_labels)+1, max(predicted_labels)+1)
    if verbose:
        n_perm = np.math.factorial(k_pred)
        counter = 0
        subcounter = 1
        print(f"Number of permutations: {n_perm}")
        print(f"Number of communities: {k_pred}")

    # check compatibility of true and predicted labels
    assert n == len(predicted_labels), "The number of true labels and predicted labels must be the same."
    # assert k == len(np.unique(predicted_labels)), "The number of communities of true labels and predicted labels must be the same."

    min_norm = np.inf  # Initialize min_norm to infinity
    best_permutation = None  # Initialize best_permutation to None

    # tmp
    if not all(0 <= label < k_pred for label in predicted_labels):
        raise ValueError("All labels in 'predicted_labels' must be within the range [0, k_pred-1].")
    
    # Loop over all permutations of labels
    for permutation in itertools.permutations(range(k_pred)):
        if verbose:
            counter += 1
            if counter > subcounter * n_perm / 100:
                print(f"Progress: {np.round((counter/n_perm)*100, 2)}%")
                subcounter += 1
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


def compute_modularity(A, z):
    """
    Compute the modularity of a network given its adjacency matrix and node assignments.

    Parameters:
    A (numpy.ndarray): The adjacency matrix of the network.
    z (list): The node assignment vector.

    Returns:
    float: The modularity of the network.

    """
    # Convert the adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(np.array(A))

    # Convert the assignment vector to a partition
    partition = [set([idx for idx, value in enumerate(z) if value == class_]) for class_ in set(z)]

    # Compute the modularity of the network
    Q = modularity(G, partition)

    return Q


def entropy(z):
    """
    Calculate the entropy of a community assignment vector z.
    Communities are assumed to be labeled from 0 to k-1, although some may be missing in z.
    """
    n = len(z)
    c = Counter(z) # dictionary {label:count}
    return -sum((c[i]/n)*log2(c[i]/n) for i in c)


############################################################################################################
# Greedy loss (implementation of the loss in an approximate, greedy way that scales quadratically in the number of communities)
# auxiliary functions for the greedy loss

def switch_labels(v, l1, l2):
    for i in range(len(v)):
        if v[i] == l1:
            v[i] = l2
        elif v[i] == l2:
            v[i] = l1

def communities_sorted_by_size(v, descending = True):
    unique, counts = np.unique(v, return_counts=True)
    if descending:
        return unique[np.argsort(counts)[::-1]]
    else:
        return unique[np.argsort(counts)]

def find_max_overlap(v1, index_c1, v2, i):
    # find the index of the community in v2 having the most nodes overlapping with community c1 in v1
    # i specifies that we can already rule out the first i communities in v2
    max_overlap = 0
    max_index = -1
    k2 = len(np.unique(v2))
    v1_copy = np.array(v1.copy())
    v2_copy = np.array(v2.copy())
    for j in range(i, k2):
        overlap = np.sum((v1_copy == index_c1) & (v2_copy == j))
        if overlap > max_overlap: #strictly greater means that, in cases of draw, the community with smallest index is chosen (arbitrary)
            max_overlap = overlap
            max_index = j

    return max_index

def greedy_loss(a, b):
    # given two lists of assignments
    # checks same length
    assert len(a) == len(b)
    n = len(a)
    # pick as k the minimum number of communities
    k1 = len(np.unique(a))
    k2 = len(np.unique(b))
    k = min(k1, k2)
    # let v1 be the one with less communities
    if k1 == k:
        v1 = np.array(a.copy())
        v2 = np.array(b.copy())
    else:
        v1 = np.array(b.copy())
        v2 = np.array(a.copy())

    # sort the communities by size for v1
    communities_v1 = communities_sorted_by_size(v1)
    # permute community labels of v1 following the order of communities_v1
    for i in range(n):
        v1[i] = np.where(communities_v1 == v1[i])[0][0]

    # iterate over the communities of v1 (except the last one, automatically handled)
    for i in range(k):
        switch_labels(v2, i, find_max_overlap(v1, i, v2, i))
       
    loss = np.sum(v1 != v2)/n
    return loss
