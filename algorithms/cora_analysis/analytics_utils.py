# The attribute propagation algorithm for sparse graphs
import numpy as np
import networkx as nx
import pandas as pd
import sys
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance


# Local imports
#from single.scripts.uGLAD.utils.prepare_data import eigVal_conditionNum
from scripts.helper_scripts import eig_val_condition_num


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
    print(f'Init theta: min eig val {sorted(eig, reverse=False)[0]}, cond num {cond}')
    if min(eig)<=1e-6:
        # adjust the eigenvalue
        theta += np.eye(theta.shape[-1]) * (min_eig-min(eig))
        eig, con = eigVal_conditionNum(theta)
        print(f'Adjusted theta: min eig val {sorted(eig, reverse=False)[0]}, cond num {con}')
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


def convert_theta(theta, method="exp", alpha=1):
    """Converts the partial correlation matrix to probability transition matrix

    Args:
        theta (2D np.array): matrix DxD
        method ()
        alpha (int): scaling intensity constant

    Returns:
        transition matrix or two (2D np.array): matrix DxD
    """

    if method=="exp":
        # Get thetaE_ij = e^(alpha*theta_ij) / row-sum(theta_ij)
        # element-wise exponential 
        exp_theta = np.exp(alpha*theta)
        # row-wise normalization
        row_sums = exp_theta.sum(axis=1)
        return np.array(exp_theta / row_sums), None
    elif method=="pos":
        # zero out negative entries
        mod_theta = np.where(theta<0, 0, theta)
        row_sums = mod_theta.sum(axis=1)
        return mod_theta/row_sums, None
    elif method=="posneg":
        # zero out negative entries
        pos_theta = np.where(theta<0, 0, theta)
        row_sums = pos_theta.sum(axis=1)
        pos_theta = pos_theta/row_sums        
        # zero out positive entries
        neg_theta = np.where(theta>0, 0, theta)
        neg_theta = np.where(neg_theta<0, neg_theta*(-1), neg_theta)
        row_sums = neg_theta.sum(axis=1)
        neg_theta = neg_theta/row_sums
        return pos_theta, neg_theta

def analytical_solution(thetaE, n_t, known_cat_indices, unknown_cat_indices):
    """Given the precision matrix and the nodes with 
    known categories, analytically predict the categories 
    of the other nodes using the following formula

    n_u = (I - Θe_uu)^-1 . Θe_uk . n_k

    Args:
        thetaE (2D np.array): transition matrix DxD
        n_t (pd.Dataframe): nodes x cat (distribution)
        known_cat_indices (pd.Dataframe): indices with known categories
        unknown_cat_indices (pd.Dataframe): indices with unknown categories

    Returns:
        n_t (pd.Dataframe): nodes x cat (updated distribution)
    """
    # Getting the known category distribution
    n_k = n_t.loc[known_cat_indices]
    # # Get thetaE_ij = e^theta_ij / row-sum(theta_ij)
    # # element-wise exponential 
    # exp_theta = np.exp(alpha*theta)
    # # row-wise normalization
    # row_sums = exp_theta.sum(axis=1)
    # thetaE = exp_theta / row_sums

    # Get the sub-matrices
    thetaE_uu = thetaE[unknown_cat_indices][:, unknown_cat_indices]
    thetaE_uk = thetaE[unknown_cat_indices][:, known_cat_indices]
    t1 = np.eye(thetaE_uu.shape[0]) - thetaE_uu
    t2 = np.matmul(thetaE_uk, np.array(n_k))
    n_u = np.matmul(np.linalg.inv(t1), t2)
    # updating the distribution of the unknown categories
    n_t.loc[unknown_cat_indices] = n_u
    return n_t


def iterative_solution(thetaP, thetaN, n_t, known_cat_indices, unknown_cat_indices, epsilon=1e-3, alpha=1, norm=None, max_iter=50):
    """Given the precision matrix and the nodes with 
    known categories, runs an iterative procedure to
    predict the categories of the other nodes.

    Args:
        thetaP (2D np.array): transition matrix DxD
        theatN (2D np.array): second transition matrix DxD (optional)
        n_t (pd.Dataframe): nodes x cat (distribution)
        known_cat_indices (pd.Dataframe): indices with known categories
        unknown_cat_indices (pd.Dataframe): indices with unknown categories
        epsilon (float): the convergence threshold
        alpha (float): scaling intensity parameter
        norm (string): normalization method
        max_iter (int): maximum iterations

    Returns:
        n_t (pd.Dataframe): nodes x cat (updated distribution)
    """
    def converged(n_t, n_tm1):
        l2_norm_diff = np.linalg.norm(n_t-n_tm1)
        return l2_norm_diff <= epsilon
    # # Get thetaE_ij = e^theta_ij / row-sum(theta_ij)
    # # element-wise exponential 
    # exp_theta = np.exp(alpha*theta)
    # # row-wise normalization
    # row_sums = exp_theta.sum(axis=1)
    # thetaE = exp_theta / row_sums
    # Get the thetaE_U
    thetaP_u = thetaP[unknown_cat_indices]
    if thetaN is not None:
        thetaN_u = thetaN[unknown_cat_indices]
    # Get the unknown nodes distribution
    nu = n_t.loc[unknown_cat_indices]
    nu0 = nu
    itr = 0
    while True:
        nu_tm1 = nu.copy()
        n_t.loc[unknown_cat_indices] = nu
        # Getting the current unknown category distribution
        nu = np.matmul(thetaP_u, np.array(n_t))
        if thetaN is not None:
            nu_neg = np.matmul(thetaN_u, np.array(n_t))
        # print(nu)
        if norm=="KL":
            for i in range(nu.shape[0]):
                if nu[i,:].sum() > 0:
                    nu[i,:] += + rel_entr(np.array(nu0)[i,:], nu[i,:]) * nu[i,:]  
                if thetaN is not None:
                    # nu[i,:] = nu[i,:] - nu_neg[i,:] + rel_entr(np.array(nu0)[i,:], nu_neg[i,:]) * nu_neg[i,:]
                    if nu_neg[i,:].sum() > 0:
                        nu[i,:] += rel_entr(np.array(nu0)[i,:], nu_neg[i,:]) * nu_neg[i,:]
                if any(e < 0 for e in nu[i,:]):
                    for e in range(0, len(nu[i,:])):
                        if nu[i,e] < 0:
                            nu[i,e] = 0.000001
                # renormalizing doesn't help
                # for e in range(len(nu[i,:])):
                #     nu[i,e] = nu[i,e]/nu[i,:].sum()
        elif norm=="Wasserstein":
            for i in range(nu.shape[0]):
                nu[i,:] = nu[i,:] + wasserstein_distance(np.array(nu0)[i,:], nu[i,:]) * nu[i,:]
        elif norm is None:
            if thetaN is not None:
                nu -= nu_neg
                # renormalizing doesn't help
                # for i in range(nu.shape[0]):
                #     for e in range(len(nu[i,:])):
                #         nu[i,e] = nu[i,e]/nu[i,:].sum()
        else:
            print(f"Normalization {norm} not implemented yet")
        # print(nu)
        itr += 1
        if converged(nu, nu_tm1) or (itr > max_iter): 
            #print(f'Iterative algorithm ran for iterations={itr}')
            # Adding, otherwise we miss the last update
            n_t.loc[unknown_cat_indices] = nu
            break
    return n_t  


def neighbor_vote(theta, n_t, known_cat_indices, unknown_cat_indices, epsilon=0.1, max_iter=35):  #1e-1):
    """Given the precision matrix and the nodes with 
    known categories, runs an iterative procedure to
    predict the categories of the other nodes.

    Args:
        theta (2D np.array): precision matrix DxD
        n_t (pd.Dataframe): nodes x cat (distribution)
        known_cat_indices (pd.Dataframe): indices with known categories
        unknown_cat_indices (pd.Dataframe): indices with unknown categories
        epsilon (float): the convergence threshold
        max_iter (int):  max iterations

    Returns:
        n_t (pd.Dataframe): nodes x cat (updated distribution)
    """
    def converged(n_t, n_tm1):
        l2_norm_diff = np.linalg.norm(n_t-n_tm1)
        return l2_norm_diff <= epsilon
    # Get the theta_U
    theta_u = theta[unknown_cat_indices]
    # Get the unknown nodes distribution
    nu = n_t.loc[unknown_cat_indices]
    itr = 0
    while True:
        nu_tm1 = nu.copy()
        n_t.loc[unknown_cat_indices] = nu
        # Getting the current unknown category distribution
        nu = np.matmul(theta_u, np.array(n_t))
        itr += 1
        if converged(nu, nu_tm1) or (itr > max_iter): 
            # print(f'Neighbor vote algorithm ran for iterations={itr}')
            # Adding, otherwise we miss the last update
            n_t.loc[unknown_cat_indices] = nu
            break
    return n_t  





def propagate_attribute(theta, node_attribute_dict, unknown_cat='u', method='analytical', alpha=1, norm=None, conv_method="exp", max_iter=50):
    """Algorithm to run the attribute/label propagation
    among the nodes based on the input precision matrix theta. 
    Returns an updated dictionary with predicted catergories
    for the remaining nodes.

    Args:
        theta (2D np.array): precision matrix DxD
        node_attribute_dict (dict): {'name':'category'}
        unknown_cat (str): The marker for the unknown category
        method (str): analytical/iterative
        alpha (float): scaling intensity parameter
        norm (string): normalization method for the iterative solution
        conv_method (string): the method used to conver the partial correlation matrix 
                              to probability transition matrix (or matrices)
        max_iter (int): maximum number of iterations

    Returns:
        predicted_categories (dict): {'name':'category'}
        n_t (pd.Dataframe): nodes x cat (updated distribution)
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
    #print(f'Running the {method} method')
    # Get the updated distribution over the categories
    if method=='analytical':
        mod_theta, _ = convert_theta(theta, method=conv_method, alpha=alpha)
        n_t = analytical_solution(mod_theta, n_t, known_cat_indices, unknown_cat_indices)
    elif method=='iterative':
        thetaP, thetaN = convert_theta(theta, method=conv_method, alpha=alpha)
        n_t = iterative_solution(thetaP, thetaN, n_t, known_cat_indices, unknown_cat_indices, alpha=alpha, norm=norm, max_iter=max_iter)
    elif method=='neighbor_vote':
        n_t = neighbor_vote(theta, n_t, known_cat_indices, unknown_cat_indices, max_iter=max_iter)
    else:
        print(f'{method} method not valid. Enter analytical/iterative')
        sys.exit(0)
    # Assigning the category with the highest probability
    predicted_categories = n_t.idxmax(axis=1)
    # print(f'Prediction:\nDistribution{n_t}\n\nCategories{predicted_categories.to_dict()}')
    return predicted_categories, n_t