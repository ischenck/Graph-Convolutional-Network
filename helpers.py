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
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1, where=rowsum!=0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)

def normalize_adj(adj: 'scipy.sparse.csr_matrix') -> 'scipy.sparse.csr_matrix':
    """ symmetrically normalize adjacency matrix

    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt * adj * d_mat_inv_sqrt

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

def save_csr_matrix(file: 'pathlib.Path', csr_mx: 'scipy.sparse.csr_matrix') -> None:
    """ Saves sparse csr as file

    Args:
        file = PathLib file path
        csr_mx: Compressed Sparse Row Scipy Matrix
    """
    np.savez(file, data = csr_mx.data, indices = csr_mx.indices,
        indptr = csr_mx.indptr, shape = csr_mx.shape)

def load_csr_matrix(file: 'pathlib.Path') -> 'sp.csr_matrix':
    """ Loads sparse csr file

    Args:
        file = PathLib file path

    Returns:
        Compressed Sparse Row Scipy Matrix

    """
    loader = np.load(file)
    return sp.csr_matrix((loader['data'], loader['indices'],
        loader['indptr']), shape = loader['shape']).astype(np.float16)

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
        tx_extended[index_sorted - index_sorted[0], :] = tx
        tx = tx_extended 
        ty_extended = np.zeros((len(index_range), y.shape[1]))
        ty_extended[index_sorted - index_sorted[0], :] = ty 
        ty = ty_extended 

    if dataset == 'nell.0.001' or dataset == 'nell.0.01':  
        index_range = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(index_range, index)
        tx_extended = sp.lil_matrix((len(index_range), x.shape[1]))
        tx_extended[index_sorted - index_sorted[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(index_range), y.shape[1]))
        ty_extended[index_sorted - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[index, :] = features[index_sorted, :]

        features_file = dataset + ".features.npz"
        features_path = pathlib.Path("data", features_file)
        
        if features_path.is_file():
            features = load_csr_matrix(features_path)
        else:
            if verbose:
                print("features.npz not found. Creating feature vectors for node relations.")
            features_extended = sp.hstack((features, 
                sp.lil_matrix((features.shape[0], 
                len(isolated_node_idx)))),
                dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended).astype(np.float16)
            if verbose:
                print("Saving features.npz")
            save_csr_matrix(features_path, features)
        
        features = normalize(features)
        features = sparse_mx_to_torch(features)
    else:   
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
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch(adj)

    index_train = torch.LongTensor(range(len(y)))
    index_val = torch.LongTensor(range(len(y), len(y) + 500))
    index_test = torch.LongTensor(index_sorted)
    return adj, features, labels, index_train, index_val, index_test


if __name__ == '__main__':
    import pprint 
    stuff = dict(zip(['adj', 'features', 'labels', 
    'idx_train', 'idx_val', 'idx_test'], 
        load_data('/home/ian/repos/gcn551/data', 'nell.0.001')))
    
    for key, item in stuff.items():
        stuff[key] = (item.shape)

    pprint.pprint(stuff)