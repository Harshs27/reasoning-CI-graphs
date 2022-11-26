# The attribute propagation algorithm for sparse graphs
import numpy as np
import networkx as nx
import pandas as pd
import sys

# Local imports
# from single_task.scripts.uGLAD.utils.prepare_data import eigVal_conditionNum


def eigVal_conditionNum(A):
    """Calculates the eigenvalues and the condition 
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


def get_precision_matrix_from_graph(G, min_eig=0.1):
    """Get the precision matrix from the input 
    graph G. Obtain the adjacency matrix with the
    weight information. Adjust the diagonal to 
    get the desired eigenvalue constraints 
    satisfied.

    Args:
        G (nx.Graph): Undirected graph

    Returns:
        theta (2D np.array): A DxD precision matrix
    """
    # Get the adjacency matrix from G
    theta = nx.adjacency_matrix(G).todense()
    # Analyse the eigenvalue and condition number 
    eig, cond = eigVal_conditionNum(theta)
    # print(f'Init theta: min eig val {sorted(eig, reverse=False)[0]}, cond num {cond}')
    if min(eig)<=1e-6:
        # adjust the eigenvalue
        theta += np.eye(theta.shape[-1]) * (min_eig-min(eig))
        eig, con = eigVal_conditionNum(theta)
        # print(f'Adjusted theta: min eig val {sorted(eig, reverse=False)[0]}, cond num {con}')
    return theta

def get_partial_correlations_from_graph(G, min_eig=0.1):
    """Get the partial correlation matrix from the input 
    graph G. Obtain the adjacency matrix with the weight 
    information. Add a diagonal of ones to get the rho  
    matrix

    Args:
        G (nx.Graph): Undirected graph

    Returns:
        rho (2D np.array): A DxD partial correlation matrix
    """
    # Get the adjacency matrix from G
    rho = nx.adjacency_matrix(G).todense()
    rho += np.eye(rho.shape[-1]) 
    # Analyse the eigenvalue and condition number 
    # eig, cond = eigVal_conditionNum(rho)
    # print(f'rho: min eig val {sorted(eig, reverse=False)[0]}, cond num {cond}')
    return rho


def set_node_attributes(node_attribute_dict, node_attribute_known):
    """Updates the node attributes with the known categories

    Args:
        node_attribute_dict (dict): {'name':'category'}
        node_attribute_konwn (dict): {'name':'category'}

    Returns:
        node_attribute_dict (dict): {'name':'category'}
    """
    for n, c in node_attribute_known.items():
        if n in node_attribute_dict.keys():
            node_attribute_dict[n] = c
        else:
            print(f'node {n} not found in node_attribute_dict')
    return node_attribute_dict


def analytical_solution(theta, n_t, known_cat_indices, unknown_cat_indices):
    """Given the precision matrix and the nodes with 
    known categories, analytically predict the categories 
    of the other nodes using the following formula

    n_u = (I - Θe_uu)^-1 . Θe_uk . n_k

    Args:
        theta (2D np.array): precision matrix DxD
        n_t (pd.Dataframe): nodes x cat (distribution)
        known_cat_indices (pd.Dataframe): indices with known categories
        unknown_cat_indices (pd.Dataframe): indices with unknown categories

    Returns:
        n_t (pd.Dataframe): nodes x cat (updated distribution)
    """
    # Getting the known category distribution
    n_k = n_t.loc[known_cat_indices]
    # Get thetaE_ij = e^theta_ij / row-sum(theta_ij)
    # element-wise exponential 
    exp_theta = np.exp(theta)
    # row-wise normalization
    row_sums = exp_theta.sum(axis=1)
    thetaE = exp_theta / row_sums
    # Get the sub-matrices
    thetaE_uu = thetaE[unknown_cat_indices][:, unknown_cat_indices]
    thetaE_uk = thetaE[unknown_cat_indices][:, known_cat_indices]
    t1 = np.eye(thetaE_uu.shape[0]) - thetaE_uu
    t2 = np.matmul(thetaE_uk, np.array(n_k))
    n_u = np.matmul(np.linalg.inv(t1), t2)
    # updating the distribution of the unknown categories
    n_t.loc[unknown_cat_indices] = n_u
    return n_t


def iterative_solution(theta, n_t, known_cat_indices, unknown_cat_indices, epsilon=1e-5):
    """Given the precision matrix and the nodes with 
    known categories, runs an iterative procedure to
    predict the categories of the other nodes.

    Args:
        theta (2D np.array): precision matrix DxD
        n_t (pd.Dataframe): nodes x cat (distribution)
        known_cat_indices (pd.Dataframe): indices with known categories
        unknown_cat_indices (pd.Dataframe): indices with unknown categories
        epsilon (float): the convergence threshold

    Returns:
        n_t (pd.Dataframe): nodes x cat (updated distribution)
    """
    def converged(n_t, n_tm1):
        l2_norm_diff = np.linalg.norm(n_t-n_tm1)
        return l2_norm_diff <= epsilon
    # Get thetaE_ij = e^theta_ij / row-sum(theta_ij)
    # element-wise exponential 
    exp_theta = np.exp(theta)
    # row-wise normalization
    row_sums = exp_theta.sum(axis=1)
    thetaE = exp_theta / row_sums
    # Get the thetaE_U
    thetaE_u = thetaE[unknown_cat_indices]
    # Get the unknown nodes distribution
    nu = n_t.loc[unknown_cat_indices]
    itr = 0
    while True:
        nu_tm1 = nu.copy()
        n_t.loc[unknown_cat_indices] = nu
        # Getting the current unknown category distribution
        nu = np.matmul(thetaE_u, np.array(n_t))
        itr += 1
        if converged(nu, nu_tm1): 
            print(f'Iterative algorithm ran for iterations={itr}')
            break
    return n_t


def propagate_attribute(theta, node_attribute_dict, unknown_cat='u', method='analytical'):
    """Algorithm to run the attribute/label propagation
    among the nodes based on the input precision matrix theta. 
    Returns an updated dictionary with predicted catergories
    for the remaining nodes.

    Args:
        theta (2D np.array): precision matrix DxD
        node_attribute_dict (dict): {'name':'category'}
        unknown_cat (str): The marker for the unknown category
        method (str): analytical/iterative

    Returns:
        predicted_categories (dict): {'name':'category'}
    """
    # Get all the categories as a list
    n_t = []
    node_names = []
    for _n, c in node_attribute_dict.items():
        n_t.append(c)
        node_names.append(_n)
    # Getting the distribution along the categories
    n_t = pd.get_dummies(pd.Series(n_t))
    n_t = n_t.set_index(pd.Series(node_names))
    # For the unknown categories, initialize with uniform distritbution
    total_categories = n_t.shape[1] - 1 
    known_cat_indices = n_t[unknown_cat]==0
    unknown_cat_indices = n_t[unknown_cat]==1
    n_t.loc[unknown_cat_indices] = [1/total_categories] * (total_categories + 1)
    # Dropping the unknown category
    n_t = n_t.drop(unknown_cat, axis=1)
    print(f'Running the {method} method')
    # Get the updated distribution over the categories
    if method=='analytical':
        n_t = analytical_solution(theta, n_t, known_cat_indices, unknown_cat_indices)
    elif method=='iterative':
        n_t = iterative_solution(theta, n_t, known_cat_indices, unknown_cat_indices)
    else:
        print(f'{method} method not valid. Enter analytical/iterative')
        sys.exit(0)
    # Assigning the category with the highest probability
    predicted_categories = n_t.idxmax(axis=1)
    print(f'Prediction:\nDistribution{n_t}\n\nCategories{predicted_categories.to_dict()}')
    return predicted_categories