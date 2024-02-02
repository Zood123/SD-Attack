# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components

from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')



def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def change_A(best_edges,A,return_2 = None):

    adj = A.copy().tolil()
    for edge in best_edges:
        adj[tuple(edge)] = adj[tuple(edge[::-1])] = 1 - adj[tuple(edge)]
    adj_preprocessed = preprocess_graph(adj)

    if return_2 != None:
        return adj_preprocessed,adj
    else:
        return adj_preprocessed


def coef_calculate_one(eig_vals,eig_vec,u,eigen_vals2,coef,k):

    inverse_list = np.zeros([k,len(eig_vals)])
    for i,value in enumerate(eig_vals[:k]):
        eigen_vals2_ = eigen_vals2[i]
        select_inverse = np.matmul(eig_vec[u]*eigen_vals2_,coef)
        inverse_list[i] = select_inverse
    return inverse_list



def eigen_2(eig_vals,k):
    out =[]
    for value in eig_vals[:k]:
        eigen_vals2 = m2_eigenvalues(value, eig_vals)
        out.append(eigen_vals2)

    return out



def m2_eigenvalues(chosen_value,eig_vals):
    eig_vals2 = []
    for value in eig_vals:
        if abs(value - chosen_value) <= 0.1: # 0.14
            eig_vals2.append(0)

        else:
            eig_vals2.append(1 / (value - chosen_value))
    return np.array(eig_vals2)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


# if the attacked edges are recorded in filename, use this function to read them.
def read_attack(filename):
    f = open(filename)
    lines = f.readlines()
    load_target = []
    edges = []
    for line in lines:
        a = line.split()

        load_target.append(int(a[0]))

        flag = 1
        edge_set = []

        for node in a[1:]:
            if node == '[' or node == ']':
                continue
            else:
                if flag == 1:
                    if flag == 0:
                        raise "error0!"
                    edge_set.append([int(node.strip('[').strip(']'))])
                    flag = 0
                else:
                    if flag == 1:
                        raise "error1!"
                    edge_set[-1].append(int(node.strip('[').strip(']')))
                    flag = 1
        # print(edge_set)
        edges.append(edge_set)

    f.close()

    return load_target,edges