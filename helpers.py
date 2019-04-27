import numpy as np
import pathlib
import pickle
import scipy.sparse as sp 
import networkx as nx
import torch
from typing import Tuple

if False:
    import scipy

def load_file(path: 'pathlib.Path') -> object:
    """Loads a pickled object from a file

    Args:
        path: path to file to load
    
    Returns:
        unpickeled object from file
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with path.open('rb') as fp:
        return pickle.load(fp, encoding='latin1')

def normalize(mx: 'scipy.sparse.csr_matrix') -> 'scipy.sparse.csr_matrix':
    """Normalizes a sparse scipy matrix

    Args:
        mx: Matrix to normalize

    Returns:
        normalized matrix
    """
    print(mx.shape)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)

def sparse_mx_to_torch(sparse_mx: 'scipy.sparse.csr_matrix') -> 'torch.Tensor':
    """Converts a Sparse Scipy Matrix to a Torch Tensor

    Args:
        sparse_mx: Sparse Scipy Matrix

    Returns:
        torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(directory: str, 
              dataset: str, 
              verbose: bool = False) -> Tuple['torch.Tensor', # adjacency matrix
                                              'torch.Tensor', # features matrix
                                              'torch.Tensor', # labels
                                              'torch.Tensor', # train index
                                              'torch.Tensor', # validation index
                                              'torch.Tensor', # test index
                                             ]:
    """Loads data from a dataset

    Args:
        directory: directory with all datasets to laod
        dataset: dataset to load. (cora, citeseer, pubmed)
        verbose: if True, prints loading information. Default is False.
    
    Returns:
        Tuple of data information of format (adjacency, features, labels
            train_index, validation_index, test_index)
    """
    datadir = pathlib.Path(directory)

    if not datadir.is_dir():
        raise FileNotFoundError(directory)
    
    if verbose: 
        print(f'Loading {datadir} dataset')
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    x, y, tx, ty, allx, ally, graph = map(lambda name: load_file(datadir.joinpath(f'ind.{dataset}.{name}')), names)
    index = [int(line.strip()) for line 
             in datadir.joinpath(f'ind.{dataset}.test.index').open().readlines()]
    index_sorted = np.sort(index)

    if dataset == 'citeseer':
        if verbose:
            print('Adding test data for citeseer unlabeled test nodes')
        index_range = range(index_sorted[0], index_sorted[-1] + 1)
        tx_extended = sp.lil_matrix((len(index_range), x.shape[1]))
        tx = tx_extended 
        ty_extended = np.zeros((len(index_range), y.shape[1]))
        ty_extended[index_sorted - index_sorted[0], :] = ty 
        ty = ty_extended 

    # Process features
    features = sp.vstack((allx, tx)).tolil()
    features[index, :] = features[index_sorted, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # Process labels
    labels = np.vstack((ally, ty))
    labels[index, :] = labels[index_sorted, :]
    labels = torch.argmax(torch.LongTensor(labels), dim=1)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tocoo()
    # Make the directed graph undirected
    adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch(adj)

    index_train = torch.LongTensor(range(len(y)))
    index_val = torch.LongTensor(range(len(y), len(y) + 500))
    index_test = torch.LongTensor(index)

    return adj, features, labels, index_train, index_val, index_test

