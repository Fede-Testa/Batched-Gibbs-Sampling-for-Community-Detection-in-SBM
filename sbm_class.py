import numpy as np
from utils import *

# alternative implementation: generate command as an explicit attribute to get a sampled SBM

class SBM():
    def __init__(self, n, k, P, pi=None, z_t=None, Neal=False, gamma=None):
        self.n = n
        self.k = k
        self.P = P
        self.pi = pi
        self.z_t = z_t
        self.Neal = Neal
        self.gamma = gamma

        self.__generate_sbm()
    
    def __generate_sbm(self):
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
        if self.z_t is None:
            if self.Neal:
                self.z = self.__assignments_neal()
            else:
                self.z = self.__assignments()
        else:
            self.z = self.z_t
        # Generate adjacency matrix
        self.A = self.__generate_adjacency_matrix()

    def __assignments_neal(self):
        """
        Generate random assignments of nodes to communities using the Neal's algorithm.

        Parameters:
        - n (int): Number of nodes.
        - k (int): Number of communities.
        - gamma (float, optional): The parameter of the Dirichlet distribution. If not provided, a uniform distribution is used.

        Returns:
        - z (ndarray): Array of length n containing the community assignments for each node.
        """
        if self.gamma is None:
            self.gamma = np.ones(self.k)
        z = np.zeros(self.n)
        self.pi = np.random.dirichlet(alpha=self.gamma)
        for i in range(self.n):
            z[i] = np.random.choice(a=list(range(self.k)), p=self.pi)
        return z.astype(int)
    
    def __assignments(self):
        """
        Generate random assignments of nodes to communities.

        Parameters:
        - n (int): Number of nodes.
        - k (int): Number of communities.
        - pi (ndarray, optional): assignment probabilities. If not provided, a uniform distribution is used.

        Returns:
        - z (ndarray): Array of length n containing the community assignments for each node.
        """
        if self.pi is None:
            self.pi = np.ones(self.k) / self.k
        z = np.random.choice(a=list(range(self.k)), size=self.n, p=self.pi)
        return z
    
    def __generate_adjacency_matrix(self):
        """
        Generate an adjacency matrix based on the given community assignments and edge probabilities.

        Parameters:
        z (list): List of community assignments for each node.
        P (numpy.ndarray): Probability matrix representing the edge probabilities between communities.

        Returns:
        numpy.ndarray: Adjacency matrix representing the generated graph.
        """
        n = len(self.z)
        # Initialize the adjacency matrix 
        A = np.zeros((n, n), dtype=int)
        # Generate the adjacency matrix (indexing ensures 0 diagonal elements)
        for i in range(n):
            for j in range(i+1, n):
                # Generate an edge with probability P[z[i], z[j]]
                if np.random.rand() <= self.P[self.z[i], self.z[j]]:
                    A[i, j] = 1
                    A[j, i] = 1
        return A
    
    def get_Z(self):
        return one_hot_encode(self.z)
    
    def get_A(self):
        return self.A
    
    def get_z(self):
        return self.z
    
    def get_P(self):
        return self.P
    
    def get_pi(self):
        return self.pi
    
    def get_z_t(self):
        return self.z_t
    
    def get_gamma(self):
        return self.gamma    

class Homogeneous_SBM(SBM):
    def __init__(self, n, k, p, q, pi=None, z_t=None, Neal=False, gamma=None):
        self.p = p
        self.q = q
        self.P = np.ones((k, k)) * q + np.eye(k) * (p - q)
        super().__init__(n, k, self.P, pi=pi, z_t=z_t, Neal=Neal, gamma=gamma)
