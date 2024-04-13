from tqdm import tqdm
import numpy as np
from utils import one_hot_encode
import numpy.random as rnd

class sequential_Gibbs_sampler():
    """
    A class representing a sequential Gibbs sampler for community detection in SBM.

    Parameters:
    - A: numpy array, adjacency matrix of the graph
    - z_0: numpy array, initial community assignments
    - alpha_p_pri: float, prior hyperparameter for alpha_p
    - beta_p_pri: float, prior hyperparameter for beta_p
    - alpha_q_pri: float, prior hyperparameter for alpha_q
    - beta_q_pri: float, prior hyperparameter for beta_q
    - pi_pri: numpy array, prior distribution for community assignments

    Attributes:
    - A: numpy array, adjacency matrix of the graph
    - z: numpy array, current community assignments
    - alpha_p_pri: float, prior hyperparameter for alpha_p
    - beta_p_pri: float, prior hyperparameter for beta_p
    - alpha_q_pri: float, prior hyperparameter for alpha_q
    - beta_q_pri: float, prior hyperparameter for beta_q
    - pi_pri: numpy array, prior distribution for community assignments
    - z_list: list, list of community assignments at each step
    - p_list: list, list of sampled p values at each step
    - q_list: list, list of sampled q values at each step
    - alpha_p: float, posterior hyperparameter for alpha_p
    - beta_p: float, posterior hyperparameter for beta_p
    - alpha_q: float, posterior hyperparameter for alpha_q
    - beta_q: float, posterior hyperparameter for beta_q
    - pi: numpy array, posterior distribution for community assignments
    - p: float, sampled p value
    - q: float, sampled q value
    - n: int, number of nodes in the graph
    - k: int, number of communities
    - Z: numpy array, one-hot encoded community assignments

    Methods:
    - __beta_updates: private method to update beta parameters
    - __z_updates: private method to update community assignments
    - step: perform one step of the Gibbs sampling algorithm
    - run: run the Gibbs sampler for a specified number of iterations
    - get_z_list: get the list of community assignments at each step
    - get_p_list: get the list of sampled p values at each step
    - get_q_list: get the list of sampled q values at each step
    """

    def __init__(self, A, z_0, alpha_p_pri, beta_p_pri, alpha_q_pri, beta_q_pri, pi_pri):
        # data
        self.A = A
        self.z = z_0.copy()

        # priors
        self.alpha_p_pri = alpha_p_pri
        self.beta_p_pri = beta_p_pri
        self.alpha_q_pri = alpha_q_pri
        self.beta_q_pri = beta_q_pri
        self.pi_pri = pi_pri.copy()
        #lists for analyses
        self.z_list=[self.z]
        self.p_list=[]
        self.q_list=[]

        #posteriors
        self.alpha_p = None
        self.beta_p = None
        self.alpha_q = None
        self.beta_q = None
        self.pi = self.pi_pri.copy()

        #sampled
        self.p = None
        self.q = None

        self.n = len(self.z)
        self.k = len(np.unique(self.z))

        self.Z = one_hot_encode(self.z)

    def __beta_updates(self):
        """
        Update the beta parameters based on the current community assignments.
        """
        A_lower = np.tril(self.A)
        cA_lower = np.tril(np.ones((self.n,self.n))) - np.eye(self.n) - A_lower

        U = self.Z.T @ A_lower @ self.Z
        cU = self.Z.T @ cA_lower @ self.Z

        self.alpha_p = self.alpha_p_pri + np.sum(np.diag(U))
        self.beta_p = self.beta_p_pri + np.sum(np.diag(cU))

        # set the diagonal elements to 0
        np.fill_diagonal(U, 0)
        np.fill_diagonal(cU, 0)

        self.alpha_q = self.alpha_q_pri +  np.sum(U)
        self.beta_q = self.beta_q_pri + np.sum(cU)

        self.p = np.random.beta(self.alpha_p, self.beta_p)
        self.q = np.random.beta(self.alpha_q,self.beta_q)
        self.p_list.append(self.p)
        self.q_list.append(self.q)
        return

    def __z_updates(self, append = True):
        """
        Update the community assignments based on the current beta parameters.
        """
        # compute useful quantities
        t = np.log((self.p * (1-self.q)) / ((1-self.p) * self.q))/2
        lambd = np.log((1-self.q)/(1-self.p)) / (2*t)
        lambd_mat = lambd * ( np.ones((self.n, self.n)) - np.eye(self.n) )

        for i in range(self.n):
            self.pi[i, :] = self.pi_pri[i,:] * np.exp(
                2 * t * (
                    self.A[i, :]-lambd_mat[i, :]
                    ).reshape(1, -1) @ self.Z
                    )
                    
            self.pi[i, :] = self.pi[i, :] / np.sum(self.pi[i, :])
            # sample z from pi
            self.z[i] = np.random.choice(self.k, p=self.pi[i, :])
            # update the one hot encoding of Z
            self.Z[i,:] = np.zeros(self.k)
            self.Z[i, self.z[i]] = 1
        if append:
            self.z_list.append(self.z.copy())

    def step(self, append_z = True):
        """
        Perform one step of the Gibbs sampling algorithm.

        Parameters:
        - append_z: bool, whether to append the current community assignments to the z_list

        Returns:
        None
        """
        self.__beta_updates()
        self.__z_updates(append = append_z)
    
    def run(self, num_iterations, burn_in = 0):
        """
        Run the Gibbs sampler for a specified number of iterations.

        Parameters:
        - num_iterations: int, number of iterations to run the sampler
        - burn_in: int, number of burn-in iterations to discard

        Returns:
        None
        """
        for t in tqdm(range(num_iterations)):
            self.step(append_z = (t >= burn_in))

    def get_z_list(self):
        """
        Get the list of community assignments at each step.

        Returns:
        list: list of numpy arrays representing community assignments
        """
        return self.z_list
    
    def get_p_list(self):
        """
        Get the list of sampled p values at each step.

        Returns:
        list: list of floats representing sampled p values
        """
        return self.p_list
    
    def get_q_list(self):
        """
        Get the list of sampled q values at each step.

        Returns:
        list: list of floats representing sampled q values
        """
        return self.q_list
    
##############################################################################################
##############################################################################################

class batched_Gibbs_sampler():
    """
    A class representing a batched Gibbs sampler for community detection in SBM.

    Parameters:
    - A: numpy array, adjacency matrix of the graph
    - z_0: numpy array, initial community assignment
    - alpha_p_pri: float, prior hyperparameter for alpha_p
    - beta_p_pri: float, prior hyperparameter for beta_p
    - alpha_q_pri: float, prior hyperparameter for alpha_q
    - beta_q_pri: float, prior hyperparameter for beta_q
    - pi_pri: numpy array, prior probability distribution over communities

    Attributes:
    - A: numpy array, adjacency matrix of the graph
    - z: numpy array, current community assignment
    - alpha_p_pri: float, prior hyperparameter for alpha_p
    - beta_p_pri: float, prior hyperparameter for beta_p
    - alpha_q_pri: float, prior hyperparameter for alpha_q
    - beta_q_pri: float, prior hyperparameter for beta_q
    - pi_pri: numpy array, prior probability distribution over communities
    - z_list: list, list of community assignments at each step
    - p_list: list, list of sampled p values at each step
    - q_list: list, list of sampled q values at each step
    - alpha_p: float, posterior hyperparameter for alpha_p
    - beta_p: float, posterior hyperparameter for beta_p
    - alpha_q: float, posterior hyperparameter for alpha_q
    - beta_q: float, posterior hyperparameter for beta_q
    - pi: numpy array, posterior probability distribution over communities
    - p: float, sampled p value
    - q: float, sampled q value
    - n: int, number of nodes in the graph
    - k: int, number of communities
    - Z: numpy array, one-hot encoded community assignment matrix

    Methods:
    - __init__(self, A, z_0, alpha_p_pri, beta_p_pri, alpha_q_pri, beta_q_pri, pi_pri): Initializes the batched Gibbs sampler with the given parameters
    - __beta_updates(self): Updates the posterior hyperparameters alpha_p, beta_p, alpha_q, beta_q, and samples p and q
    - __z_updates(self, append=True): Updates the community assignment z based on the current values of p, q, and pi
    - step(self, append_z=True): Performs one step of the Gibbs sampling algorithm
    - run(self, num_iterations, burn_in=0): Runs the Gibbs sampler for the specified number of iterations
    - get_z_list(self): Returns the list of community assignments at each step
    - get_p_list(self): Returns the list of sampled p values at each step
    - get_q_list(self): Returns the list of sampled q values at each step
    """

    def __init__(self, A, z_0, alpha_p_pri, beta_p_pri, alpha_q_pri, beta_q_pri, pi_pri):
        """
        Initializes the batched Gibbs sampler with the given parameters.

        Parameters:
        - A: numpy array, adjacency matrix of the graph
        - z_0: numpy array, initial community assignment
        - alpha_p_pri: float, prior hyperparameter for alpha_p
        - beta_p_pri: float, prior hyperparameter for beta_p
        - alpha_q_pri: float, prior hyperparameter for alpha_q
        - beta_q_pri: float, prior hyperparameter for beta_q
        - pi_pri: numpy array, prior probability distribution over communities
        """
        # data
        self.A = A
        self.z = z_0.copy()

        # priors
        self.alpha_p_pri = alpha_p_pri
        self.beta_p_pri = beta_p_pri
        self.alpha_q_pri = alpha_q_pri
        self.beta_q_pri = beta_q_pri
        self.pi_pri = pi_pri.copy()
        #lists for analyses
        self.z_list=[self.z]
        self.p_list=[]
        self.q_list=[]

        #posteriors
        self.alpha_p = None
        self.beta_p = None
        self.alpha_q = None
        self.beta_q = None
        self.pi = self.pi_pri.copy()

        #sampled
        self.p = None
        self.q = None

        self.n = len(self.z)
        self.k = len(np.unique(self.z))

        self.Z = one_hot_encode(self.z)

    def __beta_updates(self):
        """
        Updates the posterior hyperparameters alpha_p, beta_p, alpha_q, beta_q, and samples p and q.
        """
        A_lower = np.tril(self.A)
        cA_lower = np.tril(np.ones((self.n,self.n))) - np.eye(self.n) - A_lower

        U = self.Z.T @ A_lower @ self.Z
        cU = self.Z.T @ cA_lower @ self.Z

        self.alpha_p = self.alpha_p_pri + np.sum(np.diag(U))
        self.beta_p = self.beta_p_pri + np.sum(np.diag(cU))

        # set the diagonal elements to 0
        np.fill_diagonal(U, 0)
        np.fill_diagonal(cU, 0)

        self.alpha_q = self.alpha_q_pri +  np.sum(U)
        self.beta_q = self.beta_q_pri + np.sum(cU)

        self.p = np.random.beta(self.alpha_p, self.beta_p)
        self.q = np.random.beta(self.alpha_q,self.beta_q)
        self.p_list.append(self.p)
        self.q_list.append(self.q)
        return

    def __z_updates(self, append=True):
        """
        Updates the community assignment z based on the current values of p, q, and pi.

        Parameters:
        - append: bool, whether to append the updated community assignment to the z_list
        """
        # compute useful quantities
        t = np.log((self.p * (1-self.q)) / ((1-self.p) * self.q))/2
        lambd = np.log((1-self.q)/(1-self.p)) / (2*t)
        lambd_mat = lambd * ( np.ones((self.n, self.n)) - np.eye(self.n) )

        # batch update on pi
        self.pi = self.pi_pri * np.exp(2 * t * (self.A - lambd_mat).T @ self.Z)
        self.pi = self.pi / np.sum(self.pi, axis=1).reshape(-1, 1)

        # sample z from pi
        # using gumbel trick to exploit parallelization
        self.z = np.argmax(np.log(self.pi) + rnd.gumbel(size=self.pi.shape), axis=1)
        #self.z = np.array([np.random.choice(self.k, p=self.pi[i,:]) for i in range(self.n)])
        if append:
            self.z_list.append(self.z.copy())

    def step(self, append_z=True):
        """
        Performs one step of the Gibbs sampling algorithm.

        Parameters:
        - append_z: bool, whether to append the updated community assignment to the z_list
        """
        self.__beta_updates()
        self.__z_updates(append=append_z)
    
    def run(self, num_iterations, burn_in=0):
        """
        Runs the Gibbs sampler for the specified number of iterations.

        Parameters:
        - num_iterations: int, number of iterations to run the Gibbs sampler
        - burn_in: int, number of burn-in iterations to discard from the beginning
        """
        for t in tqdm(range(num_iterations)):
            self.step(append_z=(t >= burn_in))

    def get_z_list(self):
        """
        Returns the list of community assignments at each step.

        Returns:
        - z_list: list, list of community assignments at each step
        """
        return self.z_list
    
    def get_p_list(self):
        """
        Returns the list of sampled p values at each step.

        Returns:
        - p_list: list, list of sampled p values at each step
        """
        return self.p_list
    
    def get_q_list(self):
        """
        Returns the list of sampled q values at each step.

        Returns:
        - q_list: list, list of sampled q values at each step
        """
        return self.q_list

