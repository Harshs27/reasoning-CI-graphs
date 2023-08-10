from pyexpat import features
import numpy as np
import pandas as pd
import re
import io, os, sys
from time import time

import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
from sklearn import covariance
from scipy import linalg
from torch import kthvalue
import networkx as nx
from pprint import pprint

### ORGANISMS TABLE

# rolling up the taxonomy to the requested level
# def roll_up(string, indicator):
#     g = string.find(indicator)
#     if g == -1:
#         return string
#     end = string[g:].find(';') 
#     if end == -1:
#         end = len(string)
#     return string[:g+end].rstrip(';')


# # shortening the names
# def last_seg(x, level):
#     if x.rfind(level+'__') == -1:
#         return x
#     candidate = x[x.rfind(level+'__')+3:]
#     if candidate.find('uncultured') > -1 or candidate.find('digester') > -1 or candidate.find('bacterium_enrichment') > -1 or candidate.find('metagenome') > -1:
#         return x
#     else:
#         # for feature level, preserve the number
#         num = re.findall('^\d+', x)
#         if len(num) > 0:
#             return num[0] + '__' + candidate
#         else:
#             return candidate


# # filtering by the number of individuals in at least one sample and presence in a minimum number of samples
# def filter_by_frequency(df, min_individuals, min_samples):
#     cols_to_delete = []
#     for col in df.columns:
#         if df[col].max() < min_individuals:
#             cols_to_delete.append(col)
#         elif df[col].astype(bool).sum() < min_samples:
#             cols_to_delete.append(col)
#     df.drop(cols_to_delete, axis=1, inplace=True)
#     return df


# filtering by the number of individuals in at least one sample and presence in a minimum number of samples
# def filter_by_frequency(df, min_individuals, min_samples_present):#, min_samples):
#     # df = samples x organisms
#     # Getting the min_samples as a percentage
#     print(f'DA: Number of columns before filtering organisms table {df.shape[1]}')
#     if min_samples_present == None:
#         min_samples = np.array(df.astype(bool).sum(axis=0)) # count of each organism
#         # print(f'Input min_samples after {min_samples, len(min_samples)}')
#         # Only retaining 20% of the columns
#         # MAX_DISPLAY_NODES = 40
#         TOTAL_SAMPLES = len(min_samples)
#         kth_largest_index = int(0.20 * len(min_samples))
#         # kth_largest_index = min(MAX_DISPLAY_NODES, TOTAL_ORGANISMS)
#         print(f'DA: kth index {kth_largest_index}')
#         min_samples.sort()
#         min_samples = min_samples[-1*kth_largest_index]
#         filter_description = f'Organisms present in at least {min_samples} samples' # (equals to top 20% = {round(0.2*TOTAL_SAMPLES, 3)} samples)\n'
#     else:
#         TOTAL_SAMPLES = df.shape[0]
#         min_samples = int(TOTAL_SAMPLES * min_samples_present)
#         filter_description = f'Organisms must be present in at least {min_samples} samples ({min_samples_present} of all {TOTAL_SAMPLES} samples)'
#     # filter_description = f'Organisms count >= {min_samples} (equals to top % {round(MAX_DISPLAY_NODES/TOTAL_ORGANISMS, 3)})\n'
#     print(filter_description)
#     cols_to_delete = []
#     for col in df.columns:
#         if df[col].max() < min_individuals:
#             cols_to_delete.append(col)
#         elif df[col].astype(bool).sum() < min_samples:
#             cols_to_delete.append(col)
#     print(f'DA: NUmber of cols to be filtered out {len(cols_to_delete)}')
#     # df.drop(cols_to_delete, axis=1, inplace=True)
#     # print(f'DA: Number of columns after filtering organisms table {df.shape[1]}')
#     return cols_to_delete, filter_description


# normalize the organisms table
def norm_CLR(table):
    """Centered log Ratio: Proposed originally by Aitchison, 
    log-ratio transformations are capable of removingthe unit-sum 
    constraint of compositional data, allowing ratios to be 
    analyzed in the Euclidean spaceAitchison[1], Kurtz et al. [12].
    The Centered log-ratio (CLR) for sample (j) is

    CLR(w(j)) = [ log(w_1/g(w(j)), ..., log(w_F/g(w(j)) ]
              = log(w(j)) - log(g(w(j)))  --> Vectorized version

    where g(x) = (∏x)^(1/m) is the geometric mean of a 
    length m vector x. g(w(j)) is the geometric mean over 
    row j (over all features)

    NOTE: A pseudocount of 1 is included for calculation of 
    geometric means to avoid the undefined log[0].

    Args:
        table (pd.DataFrame): samples(rows) x features(columns)
    
    Returns:
        table (pd.DataFrame): Normalized table - samples x features
    """
    term1 = np.log(table+1)
    term2 = np.log(gmean(table+1, axis=1))  # row-wise norm
    table_clr_norm = (term1.T-term2).T
    return table_clr_norm


def normalize_org_table(table, norm):
    if norm == 'CLR':
        table = norm_CLR(table)
    elif norm == 'TSS':
        table = table.div(table.sum(axis=1), axis=0)
    else:
        print("ERROR: unknown normalization method requested: {norm}")
    return table


# Process the table for organisms at the appropriate rollup level
# def process_organisms_table(
#     organisms_table,
#     taxonomy_level, 
#     min_individuals, 
#     # min_samples, 
#     normalization
#     ):
#     """Process the organisms_table table based on the input desired parameters

#     Args:
#         organisms_table (pd.DataFrame): The organisms table 
#         taxonomy_level (str): 'feature'/'species'/'genus'/'family'
#         min_individuals (int): threshold for min number of samples
#         min_samples (int): threshold for sum of samples
#         normalization (str): CLR/TSS

#     Returns:
#         organisms_table (pd.DataFrame): The processed organisms table 
#     """
#     prefix = {'feature': 's', 'species': 's', 'genus': 'g', 'family': 'f'}
    
#     print(f"DA: The shape of the organisms table is {organisms_table.shape}")

#     # to avoid errors in groupby
#     organisms_table.fillna(0, inplace=True)
    
#     # roll up the taxonomy
#     if taxonomy_level == 'genus':
#         organisms_table['genus'] = organisms_table.Taxonomy.apply(lambda x : roll_up(x, prefix[taxonomy_level] + '__'))
#         table = organisms_table.groupby(['genus']).sum()
#     elif taxonomy_level == 'family':
#         organisms_table['family'] = organisms_table.Taxonomy.apply(lambda x: roll_up(x, prefix[taxonomy_level] + '__'))
#         table = organisms_table.groupby(['family']).sum()
#     elif taxonomy_level == 'species':
#         table = organisms_table.groupby(['Taxonomy']).sum()
#     elif taxonomy_level == 'feature':
#         # table = organisms_table.copy()
#         # table['name'] = table.apply(lambda row: str(row.name) + '__' + row.Taxonomy, axis=1) 
#         organisms_table['name'] = organisms_table.apply(lambda row: str(row.name) + '__' + row.Taxonomy, axis=1) 
#         organisms_table.set_index('name', inplace=True) 
#         table = organisms_table
#         del table['Taxonomy']
#         del table['featureID']
#     else:
#         print(f"DA: ERROR: unrecognized taxonomy level, {taxonomy_level}")
#         return
#     # transpose, shorten names and filter
#     table = table.transpose()

#     cols_to_delete, filter_description = filter_by_frequency(table, min_individuals, min_samples_present=None)#, min_samples)
#     # normalize
#     table = normalize_org_table(table, normalization)
#     table.drop(cols_to_delete, axis=1, inplace=True)
#     print(f'DA: Number of columns after filtering organisms table {table.shape[1]}')
    
#     columns = [last_seg(i, prefix[taxonomy_level]) for i in table.columns]
#     table.columns = columns

#     # if any samples were completely zero'ed out, TSS will result in NaN entries
#     table.fillna(0, inplace=True)

#     # return organism table
#     return table, filter_description


# scale conditions by columns
# def scale_df(df):
#     scaler = MinMaxScaler()
#     col = df.columns
#     index = df.index
#     scaled = scaler.fit_transform(df.values)
#     return pd.DataFrame(scaled, columns=col, index=index)


# def process_conditions_table(
#     conditions_table
#     ):
#     """
#     Process the conditions table based on the input data type

#     Args: 
#         conditions_table (pd.DataFrame): input table 

#     Returns:
#         conditions_table (pd.DataFrame): processed table 
#     """
#     # conditions_table = scale_df(conditions_table)
#     conditions_table = eliminate_cat_columns(conditions_table)
#     # conditions_table = process_table(process_table(conditions_table))
#     return conditions_table


# def combine_tables(organisms_table, conditions_table):
#     if organisms_table is None:
#         return conditions_table
#     elif conditions_table is None:
#         return organisms_table
#     else:
#         combined_table = pd.merge(organisms_table, conditions_table, left_index=True, right_index=True)
#         # combined_table = scale_df(combined_table)
#         return combined_table


def eliminate_cat_columns(df):
    cat_columns = []
    for c in df.columns:
        if df[c].dtype == 'object':
            cat_columns.append(c)
    if len(cat_columns) == 0:
        return df
    df_only_numeric = df.drop(cat_columns, axis=1)
    return df_only_numeric


# Processing the input data to be compatiable for the sparse graph recovery models
def process_table(table, NORM='no', MIN_VARIANCE=0.0, msg='', COND_NUM=70, eigval_th=1e-3):
    """Processing the input data to be compatiable for the 
    sparse graph recovery models. Checks for the following
    issues in the input tabular data (real values only).

    Note: The order is important. Repeat the function 
    twice: process_table(process_table(table)) to ensure
    the below conditions are satisfied.

    1. Remove all the rows with zero entries
    2. Fill Nans with column mean
    3. Remove columns containing only a single entry
    4. Remove columns with duplicate values
    5. Remove columns with low variance after centering

    The above steps are taken in order to ensure that the
    input matrix is well-conditioned. 

    Args:
        table (pd.DataFrame): The input table with headers
        NORM (str): min_max/mean/no
        MIN_VARIANCE (float): Drop the columns below this 
            variance threshold
        COND_NUM (int): The max condition number allowed
        eigval_th (float): Min eigval threshold. Making sure 
            that the min eigval is above this threshold by 
            droppping highly correlated columns

    Returns:
        table (pd.DataFrame): The processed table with headers
    """
    start = time()
    print(f'{msg}: Processing the input table for basic compatibility check')
    print(f'{msg}: The input table has sample {table.shape[0]} and features {table.shape[1]}')
    
    total_samples = table.shape[0]

    # typecast the table to floats
    table = table._convert(numeric=True)

    # 1. Removing all the rows with zero entries as the samples are missing
    table = table.loc[~(table==0).all(axis=1)]
    print(f'{msg}: Total zero samples dropped {total_samples - table.shape[0]}')

    # 2. Fill nan's with mean of columns
    table = table.fillna(table.mean())

    # 3. Remove columns containing only a single value
    single_value_columns = []
    for col in table.columns:
        if len(table[col].unique()) == 1:
            single_value_columns.append(col)
    table.drop(single_value_columns, inplace=True, axis=1)
    print(f'{msg}: Single value columns dropped: total {len(single_value_columns)}, columns {single_value_columns}')

    # Normalization of the input table
    table = normalize_table(table, NORM)

    # Analysing the input table's covariance matrix condition number
    analyse_condition_number(table, 'Input')
 
    # 4. Remove columns with duplicate values
    all_columns = table.columns
    table = table.T.drop_duplicates().T  
    duplicate_columns = list(set(all_columns) - set(table.columns))
    print(f'{msg}: Duplicates dropped: total {len(duplicate_columns)}, columns {duplicate_columns}')

    # 5. Columns having similar variance have a slight chance that they might be almost duplicates 
    # which can affect the condition number of the covariance matrix. 
    # Also columns with low variance are less informative
    table_var = table.var().sort_values(ascending=True)
    # print(f'{msg}: Variance of the columns {table_var.to_string()}')
    # Dropping the columns with variance < MIN_VARIANCE
    low_variance_columns = list(table_var[table_var<MIN_VARIANCE].index)
    table.drop(low_variance_columns, inplace=True, axis=1)
    print(f'{msg}: Low Variance columns dropped: min variance {MIN_VARIANCE},\
    total {len(low_variance_columns)}, columns {low_variance_columns}')

    # Analysing the processed table's covariance matrix condition number
    cov_table, eig, con = analyse_condition_number(table, 'Processed')

    itr = 1
    while con > COND_NUM: # ill-conditioned matrix
        print(f'{msg}: {itr} Condition number is high {con}. \
        Dropping the highly correlated features in the cov-table')
        # Find the number of eig vals < eigval_th for the cov_table matrix.
        # Rough indicator of the lower bound num of features that are highly correlated.
        eig = np.array(sorted(eig))
        lb_ill_cond_features = len(eig[eig<eigval_th])
        print(f'Current lower bound on ill-conditioned features {lb_ill_cond_features}')
        if lb_ill_cond_features == 0:
            print(f'All the eig vals are > {eigval_th} and current cond num {con}')
            if con > COND_NUM:
                lb_ill_cond_features = 1
            else:
                break
        highly_correlated_features = get_highly_correlated_features(cov_table)
        # Extracting the minimum num of features making the cov_table ill-conditioned
        highly_correlated_features = highly_correlated_features[
            :min(lb_ill_cond_features, len(highly_correlated_features))
        ]
        # The corresponding column names
        highly_correlated_columns = table.columns[highly_correlated_features]
        print(f'{msg} {itr}: Highly Correlated features dropped {highly_correlated_columns}, \
        {len(highly_correlated_columns)}')
        # Dropping the columns
        table.drop(highly_correlated_columns, inplace=True, axis=1)
        # Analysing the processed table's covariance matrix condition number
        cov_table, eig, con = analyse_condition_number(table, f'{msg} {itr}: Corr features dropped')
        # Increasing the iteration number
        itr += 1
    print(f'{msg}: The processed table has sample {table.shape[0]} and features {table.shape[1]}')
    print(f'{msg}: Total time to process the table {np.round(time()-start, 3)} secs')
    return table


def get_highly_correlated_features(input_cov):
    """Taking the covariance of the input covariance matrix
    to find the highly correlated features that makes the 
    input cov matrix ill-conditioned.

    Args:
        input_cov (2D np.array): DxD matrix

    Returns:
        features_to_drop (np.array): List of indices to drop
    """
    cov2 = covariance.empirical_covariance(input_cov)
    # mask the diagonal 
    np.fill_diagonal(cov2, 0)
    # Get the threshold for top 10% 
    cov_upper = upper_tri_indexing(np.abs(cov2))
    sorted_cov_upper = [i for i in sorted(enumerate(cov_upper), key=lambda x:x[1], reverse=True)]
    th = sorted_cov_upper[int(0.1*len(sorted_cov_upper))][1]
    # Getting the feature correlation dictionary
    high_indices = np.transpose(np.nonzero(np.abs(cov2) >= th))
    high_indices_dict = {}
    for i in high_indices: # the upper triangular part
        if i[0] in high_indices_dict:
            high_indices_dict[i[0]].append(i[1])
        else:
            high_indices_dict[i[0]] = [i[1]]
    # sort the features based on the number of other correlated features.
    top_correlated_features = [[f, len(v)] for (f, v) in high_indices_dict.items()]
    top_correlated_features.sort(key=lambda x: x[1], reverse=True)
    top_correlated_features = np.array(top_correlated_features)
    features_to_drop = top_correlated_features[:, 0] 
    return features_to_drop


def analyse_condition_number(table, MESSAGE=''):
    S = covariance.empirical_covariance(table, assume_centered=False)
    eig, con = eig_val_condition_num(S)
    print(f'{MESSAGE} covariance matrix: The condition number {con} and min eig {min(eig)} max eig {max(eig)}')
    return S, eig, con
     

def eig_val_condition_num(A):
    """Calculates the eigenvalues and the condition 
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


def normalize_table(df, typeN):
    if typeN == 'min_max':
        return (df-df.min())/(df.max()-df.min())
    elif typeN == 'mean':
        return (df-df.mean())/df.std()
    else:
        print(f'No Norm applied : Type entered {typeN}')
        return df


### uGLAD model helper functions

#  Plot covariance and precision
def plot_uGLAD_results(emp_cov, model):

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    covs = [("Empirical", emp_cov),
                ("uGLAD", model.covariance_)]
    vmax = model.covariance_.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 4, i+1)
        plt.imshow(this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title("%s covariance" % name)

    precs = [("Empirical", linalg.inv(emp_cov)),
            ("uGLAD", model.precision_)]
    vmax = 0.9 * model.precision_.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 4, i+5)
        plt.imshow(np.ma.masked_equal(this_prec, 0), interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title("%s precision" % name)


# determining the threshold to maintain the sparsity level of the graph
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]


def precision_empty(A):
    prec_upper = upper_tri_indexing(np.abs(A))
    non_zeros = np.count_nonzero(prec_upper)
    print(f"DA: number of nonzero entries in upper triangular precision matrix is {non_zeros}")
    if non_zeros == 0:
        return True
    else:
        return False


def get_partial_correlations(precision):
    """Get the partial correlation matrix from the 
    precision matrix. It applies the following 
    
    Formula: rho_ij = -p_ij/sqrt(p_ii, p_jj)
    
    Args:
        precision (2D np.array): The precision matrix
    
    Returns:
        rho (2D np.array): The partial correlations
    """
    precision = np.array(precision)
    D = precision.shape[0]
    rho = np.zeros((D, D))
    for i in range(D): # rows
        for j in range(D): # columns
            if i==j: # diagonal elements
                rho[i][j] = 1
            elif j < i: # symmetric
                rho[i][j] = rho[j][i]
            else: # i > j
                num = -1*precision[i][j]
                den = np.sqrt(precision[i][i]*precision[j][j])
                rho[i][j] = num/den
    return rho

# Plot the graph
def uGLAD_graph(
    data, 
    prec, 
    sparsity=0.1, 
    type='both', 
    title='', 
    intensity=3, 
    fig_size=12, 
    PLOT=True,
    save_file=None,
    roundOFF=3,
    plot_edge_labels=True
    ):
    G = nx.Graph()
    col = data.columns
    prec_upper = upper_tri_indexing(np.abs(prec))
    print(f"Number of nonzero entries in upper triangular precision matrix is {np.count_nonzero(prec_upper)}")
    num_non_zeros = int(sparsity*len(prec_upper))
    prec_upper.sort()
    th = prec_upper[-num_non_zeros]
    print(f'Sparsity {sparsity} using threshold {th}')
    th1, th2 = th, -1*th
    # getting the max weight and adjusting its intensity of visualization
    # NOTE: No weight scaling
    max_wt = 1 # prec_upper[-1] / intensity
    # print(f'max_wt in prec {max_wt, prec_upper, prec}')
    factor = 1  # for scaling edge weights
    graph_edge_list = []
    for i in range(len(col)):
        for j in range(i+1, len(col)):
            if (prec[i,j] > th1) and (type=='both' or type=='pos') :
                # G.add_edge(col[i], col[j], color='green', weight=round(prec[i,j]/max_wt, roundOFF), label=round(prec[i,j]/(factor*prec_upper[-1]), roundOFF))
                # _edge = '('+str(col[i])+', '+str(col[j])+', '+str(round(prec[i,j]/(factor*prec_upper[-1]), roundOFF))+', green)'
                G.add_edge(col[i], col[j], color='green', weight=round(prec[i,j]/max_wt, roundOFF), label=round(prec[i,j], roundOFF))
                _edge = '('+str(col[i])+', '+str(col[j])+', '+str(round(prec[i,j], roundOFF))+', green)'
                graph_edge_list.append(_edge)
            elif (prec[i,j] < th2) and (type=='both' or type=='neg'):
                # G.add_edge(col[i], col[j], color='red', weight=round(prec[i,j]/max_wt, roundOFF), label=round(prec[i,j]/(factor*prec_upper[-1]), roundOFF))
                # _edge = '('+str(col[i])+', '+str(col[j])+', '+str(round(prec[i,j]/(factor*prec_upper[-1]), roundOFF))+', red)'
                G.add_edge(col[i], col[j], color='red', weight=round(prec[i,j]/max_wt, roundOFF), label=round(prec[i,j], roundOFF))
                _edge = '('+str(col[i])+', '+str(col[j])+', '+str(round(prec[i,j], roundOFF))+', red)'
                graph_edge_list.append(_edge)

    if PLOT: 
        print(f'We have {len(graph_edge_list)} edges')
        # print(f'DA: graph edges {graph_edge_list}')

    edge_colors = [G.edges[e]['color'] for e in G.edges]
    edge_width = [abs(G.edges[e]['weight']) for e in G.edges]
    # Scaling the intensity of the edge_weights for viewing purposes
    # print(np.max(np.abs(edge_width)))
    if len(edge_width) > 0:
        edge_width = edge_width/np.max(np.abs(edge_width))*intensity
    edge_labels = dict([((n1, n2), G.edges[(n1, n2)]["label"]) for n1, n2 in G.edges])
    image_bytes = None
    if PLOT:
        fig = plt.figure(1, figsize=(fig_size,fig_size))
        plt.title(title)
        n_edges = len(G.edges)
        # pos = nx.spring_layout(G, scale=0.2, k=1/np.sqrt(n_edges+10))
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato') #'fdp', 'sfdp', 'neato'
        nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=100)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width)
        if plot_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        y_off = 0.008
        nx.draw_networkx_labels(G, pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()})
        # nx.draw_networkx(G, with_labels=True, edge_color=edge_colors, width=edge_width, node_color='grey', node_size=80)
        plt.title(f'{title}', fontsize=20)
        plt.margins(0.15)
        plt.tight_layout()
        
        # saving the file
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
            
        # Saving the figure in-memory
        buf = io.BytesIO()
        plt.savefig(buf)
        
        # getting the image in bytes
        buf.seek(0)
        image_bytes = buf.getvalue() # Image.open(buf, mode='r')
        buf.close()
        
        # closing the plt
        plt.close(fig)
    return G, image_bytes, graph_edge_list


def plot_graph_with_filter(
    G, 
    filter = [], 
    pos=None, 
    title='', 
    intensity=3,
    fig_num=1, 
    fig_size=12, 
    save_fig=None, 
    roundOFF=5
    ):
    """Only keep the edges where at least one node matches the filter list.
    If filter list is empty, do not filter"""
    remove_edges = [] 
    # lower case all the filter entries
    filter = [f.lower() for f in filter]
    for e in G.edges(data=True):
        # if filter not in [e[0], e[1]]:
        if e[0].lower() not in filter and e[1].lower() not in filter and len(filter)>0:
            remove_edges.append(e)
        # elif float(e[2]['weight']) < 0.05:
        #     remove_edges.append(e)
    G.remove_edges_from(list(remove_edges))
    # remove isolated nodes with no edge connections
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    graph_edge_list = []
    for e in G.edges(data=True):
        _edge = '('+str(e[0])+', '+str(e[1])+', '+str(round(e[2]["label"], roundOFF))+', '+str(e[2]["color"]+')')
        graph_edge_list.append(_edge)
    print(f'DA: we have {len(graph_edge_list)} edges in filtered graph')
    print(f'DA: filtered-node graph edges {graph_edge_list}')
    edge_colors = [G.edges[e]['color'] for e in G.edges]
    edge_width = [abs(G.edges[e]['weight']) for e in G.edges]
    # Scaling the intensity of the edge_weights for viewing purposes
    if len(edge_width) > 0:
        edge_width = edge_width/np.max(np.abs(edge_width))*intensity
    edge_labels = dict([((n1, n2), G.edges[(n1, n2)]["label"]) for n1, n2 in G.edges])

    fig = plt.figure(fig_num, figsize=(fig_size, fig_size))
    plt.title(title)
    n_edges = len(G.edges)
    if pos is None:
        # pos = nx.spring_layout(G, scale=0.2, k=1/np.sqrt(n_edges+10))
        pos = nx.nx_agraph.graphviz_layout(G, prog='neato') #'fdp', 'sfdp', 'neato'
    nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=100)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    y_off = 0.008
    nx.draw_networkx_labels(G, pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()})
    if save_fig: plt.savefig(save_fig, bbox_inches='tight')
    # Saving the figure in-memory
    buf = io.BytesIO()
    plt.savefig(buf)
    # getting the image in bytes
    buf.seek(0)
    image_bytes = buf.getvalue() # Image.open(buf, mode='r')
    buf.close()
    # closing the plt
    plt.close(fig)
    return G, image_bytes, graph_edge_list


# from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def select_best(source_array, num_points, decreasing=True):
    num_points = min(num_points, len(source_array)-1)
    source_array = np.array(source_array)
    if decreasing:
        id_array = np.argpartition(source_array, -num_points)[-num_points:]
        id_array = id_array[np.argsort(source_array[id_array])][::-1]
    else:
        id_array = np.argpartition(source_array, num_points)[0:num_points]
        id_array = id_array[np.argsort(source_array[id_array])]
    return id_array
  

def find_closest(input, col_feature_names, num_options=3):
    distances = []
    for col in col_feature_names:
        distances.append(levenshtein(input, col))
    indices = select_best(distances, num_options, decreasing=False)
    return np.array(col_feature_names)[indices]


# def shift_methane_gases(table, shift_weeks):
#     gas = ['methane', 'ch4', 'co2', 'n2-o2', 'h2s']
#     col_to_shift = []
#     for col in table.columns:
#         if np.array([col.lower().find(g) for g in gas]).max() > -1:
#             col_to_shift.append(col)
#     for col in col_to_shift:
#         table[col] = table[col].shift(shift_weeks)
#     table = table[shift_weeks:]
#     return table