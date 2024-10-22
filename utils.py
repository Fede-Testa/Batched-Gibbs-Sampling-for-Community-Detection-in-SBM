import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


def one_hot_encode(x, k=None):
    """
    One-hot encodes an array of labels.

    Parameters:
    - x (array-like): The input array of labels.
    - k (int): The number of classes. If None, the number of classes is inferred from the input array.

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
    if k is None:
        # Map the labels to integers starting from 0
        unique_labels, x_int = np.unique(x, return_inverse=True)
        dim = len(unique_labels)
        X = np.zeros((len(x_int), dim))
        X[np.arange(len(x_int)), x_int] = 1
        X = X.astype(int)
    else:
        if np.max(x) >= k:
            raise ValueError("The maximum label in the input array is greater \
                             than the number of communities.")
        X = np.zeros((len(x), k))
        X[np.arange(len(x)), x] = 1
    return X


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


def majority_vote(z_list, late_start=0):
    """
    Perform a majority vote on a list of community assignments.

    Parameters:
    - z_list (list): List of community assignments.
    - late_start (int): Number of last assignments to consider.

    Returns:
    - z_maj (ndarray): Majority votes.
    """
    z_list = np.array(z_list[-late_start:])
    z_maj = []
    for i in range(z_list.shape[1]):
        z_maj.append(np.argmax(np.bincount(z_list[:, i]))) # columnwise majority vote
    return z_maj


def salso(clusterings_list):
    """
    Finds a point estimate using posterior clustering samples.

    Parameters:
    - clusterings_list (list): List of community assignments.

    Returns:
    - result_np (ndarray): Point Estimate.
    """
    numpy2ri.activate()

    # Import the salso package
    salso = importr('salso')

    # Convert the numpy matrix to an R matrix
    r_matrix = numpy2ri.py2rpy(np.array(clusterings_list))

    # Define the R function
    r_code = """
    myfun <- function(mat) {
      library(salso)
      results <- salso(mat, maxZealousAttempts=10000)
      return(as.vector(results))
    }
    """

    # Load the R function into the R environment
    robjects.r(r_code)

    # Get the R function
    r_myfun = robjects.globalenv['myfun']

    # Call the R function with the R matrix
    result = r_myfun(r_matrix)

    # Convert the result back to a numpy array
    result_np = numpy2ri.rpy2py(result)
    
    return result_np


def progressive_R_hat(estimand_array, splits=2, n_chains=4, n_iter=250, burn_in=0, ignore=0):
    """
    Compute the progressive R_hat statistic over a series of iterations, checking
    for convergence as more samples are added.

    Args:
        estimand_array (ndarray): 2D array of shape (iterations, chains) containing samples.
        splits (int): Number of splits for each chain to compute R_hat. Defaults to 2.
        n_chains (int): Number of chains in the estimand_array. Defaults to 4.
        n_iter (int): Total number of iterations (rows) in the estimand_array.
        burn_in (int): Number of initial iterations to discard. Defaults to 0.
        ignore (int): Proportion of samples to ignore. Defaults to 0.

    Returns:
        R_hat_list (ndarray): Array of R_hat values computed for different segments.
    """

    # Allocate space for the R_hat values (indices divided by `splits`)
    R_hat_list = np.zeros((n_iter-burn_in)//splits)

    if not isinstance(estimand_array, np.ndarray):
        estimand_array = np.array(estimand_array).T

    # Iterate through the rows, taking steps of `splits`
    for r in range(1+splits, n_iter-burn_in, splits):
        l = int(r*ignore)  # Compute the initial index to ignore a portion of the samples
        # Adjust `l` to ensure (r+1-l) is a multiple of `splits`
        while ((r+1-l) % splits) or (l!=0) != 0:
            l -= 1

        # Extract the first `r+1` rows, while possibly ignoring some samples
        list_split = estimand_array[l:r+1, :]
        # Convert the 2D array into a list of arrays (one for each chain)
        list_split = [list_split[:, i] for i in range(n_chains)]
        # Split the chains into `splits` segments
        list_split = split_chains(list_split, splits)

        # Calculate and store R_hat
        R_hat_list[r//splits] = R_hat(list_split)

    return R_hat_list


def split_chains(chains, splits = 2):
    # chains is a list of arrays
    n_tot = len(chains[0])
    if n_tot % splits != 0:
        raise ValueError("The number of samples is not divisible by the number of splits")
    # split each chain into splits parts
    split_chains = []
    for chain in chains:
        split_chain = np.array_split(chain, splits)
        split_chains.append(split_chain)
    return np.vstack(split_chains).T # row index: iteration, col_index: chain


def R_hat(split_chains):
    n = split_chains.shape[0]
    m = split_chains.shape[1]

    # Verifica che ci siano abbastanza campioni per il calcolo
    if n <= 1:
        raise ValueError(f"Insufficient number of iterations per split ({n}). Increase the number of iterations or reduce the number of splits.")

    # within chains variance
    W = np.mean(np.sum((split_chains - np.mean(split_chains, axis=0))**2, axis=0)/(n-1))
    # between chains variance
    B = n * np.sum((np.mean(split_chains, axis=0) - np.mean(split_chains))**2) / (m-1)
    # estimated variance
    var_plus = (n-1)/n * W + 1/n * B

    return np.sqrt(var_plus/W)


def find_convergence(list, lb, ub):
    candidates = []
    for i in range(len(list)):
        if list[i] > lb and list[i] < ub:
            candidates.append(i)
    for i in candidates:
        # check if all the following values are within the bounds
        if all([list[j] > lb and list[j] < ub for j in range(i+1, len(list))]):
            return i
    return -1

# def one_hot_encode(x, k=4):
#     """
#     One-hot encodes an array of k labels.

#     Parameters:
#     - x (array-like): The input array of labels.
#     - k (int): The number of classes.

#     Returns:
#     - X (ndarray): The one-hot encoded array.

#     Example:
#     >>> labels = [0, 1, 2, 1, 0]
#     >>> one_hot_encode(labels, 3)
#     array([[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [0, 1, 0],
#         [1, 0, 0]])
#     """
#     X = np.zeros((len(x), k))
#     X[np.arange(len(x)), x] = 1
#     return X