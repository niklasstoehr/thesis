## Library Dependencies

import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
from networkx.generators import random_graphs


## Sorting rows by node degree (decreasing)

def sort_adjacency(g):
    node_k1 = dict(g.degree())  # sort by degree
    node_k2 = nx.average_neighbor_degree(g)  # sort by neighbor degree
    node_closeness = nx.closeness_centrality(g)
    node_betweenness = nx.betweenness_centrality(g)

    node_sorting = list()

    #for node_id in range(0, len(g)):
    for node_id in g.nodes():
        node_sorting.append(
            (node_id, node_k1[node_id], node_k2[node_id], node_closeness[node_id], node_betweenness[node_id]))

    node_descending = sorted(node_sorting, key=lambda x: (x[1], x[2]), reverse=True)

    mapping = dict()

    for i, node in enumerate(node_descending):
        mapping[node[0]] = i

    a = nx.adjacency_matrix(g, nodelist=mapping.keys()).todense()
    #g = nx.relabel_nodes(g, mapping)  # change node_id according to ordering

    return g, a


## obtain the upper triangle of adjacency matrix without diagonal

def reshape_A(a, diag_offset, node_features=None):
    if diag_offset >= 0:  ## obtain upper triangle of graph

        upper_a = np.ravel(a[np.triu_indices(a.shape[0], k=diag_offset)])

    else:  ## keep entire adjacency matrix
        upper_a = np.reshape(a, (a.shape[0], a.shape[0], 1))  ## reshape to add channel 1

    return upper_a


## calculate the area of upper triangle of sorted adjacency matrix

def calculate_A_shape(n, diag_offset):
    if diag_offset == 0:
        return (int(((n * n) / 2) + (n / 2)),)

    elif diag_offset == 1:
        return (int(((n * n) / 2) - (n / 2)),)

    elif diag_offset == -1:  # keep entire adjacency matrix
        return (n, n, 1)

    ## construct adjacency matrix from flattened upper_triangle


def reconstruct_adjacency(upper_a, clip, diag_offset):
    ## if first element is 1 --> squeeze it
    if upper_a.shape[0] == 1:
        upper_a = np.squeeze(upper_a)

    ## from upper triangle to adjacency matrix
    if diag_offset == 1:
        a = squareform(upper_a, force='no', checks=True)

    elif diag_offset == 0:
        n = int(-1 + np.sqrt(1 + 8 * len(upper_a))) // 2
        iu1 = np.triu_indices(n)
        a = np.empty((n, n))
        a[iu1] = upper_a
        a.T[iu1] = upper_a

    else:
        a = upper_a

    ## clip values to binary
    if clip == True:
        a[a >= 0.5] = 1
        a[a < 0.5] = 0

    return a


## pad the adjacency matrix to adhere to fixed size n_max

def pad_matrix(a, n_max, fill_diag):
    ## fill the diagonal with fill_diag
    np.fill_diagonal(a, fill_diag)

    max_adjacency = np.zeros([n_max, n_max])
    max_adjacency[:a.shape[0], :a.shape[1]] = a

    return max_adjacency


## unpad the adjacency matrix by looking at diagonal values

def unpad_matrix(max_adjacency, diag_value, fix_n):
    if fix_n == False:

        keep = list()
        for i in range(0, max_adjacency.shape[0]):
            if max_adjacency[i][i] == diag_value:
                keep.append(i)

        ## delete rows and columns
        max_adjacency = max_adjacency[:, keep]  # keep columns
        max_adjacency = max_adjacency[keep, :]  # keep rows

    return max_adjacency


def prepare_in_out(T, diag_offset, A_shape):
    T = np.asarray(T)

    if diag_offset >= 0:  # vector input
        return T, (A_shape[0],), A_shape[0]
    else:  # matrix input
        return T, (A_shape[0], A_shape[1], 1), (A_shape[0], A_shape[1], 1)


## Preprocess Graph____________________________________

if __name__ == "__main__":

    g = random_graphs.erdos_renyi_graph(4, 0.2, seed=None)  # no edges are removed
    g, a = sort_adjacency(g)
    upper_a = reshape_A(a, diag_offset=0)
    reconstructed_a = reconstruct_adjacency(upper_a, clip=True, diag_offset=0)

    max_adjacency = pad_matrix(a, 50, 1)
    a = unpad_matrix(max_adjacency, 1, False)

