from __future__ import absolute_import
import scipy.io as scio
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pairs import pair
import os, sys
from dgl import DGLGraph
import dgl


def load_data(args):
    if args.dataset == '3sources':
        data = ThreesourcesDataset(mi=args.mi, k=args.k)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    return data


class ThreesourcesDataset(object):

    def __init__(self, mi, k):
        self.k = k
        self.mi = mi
        self.name = '3sources'
        self.dataFile = './data/3sources.mat'
        self.data = scio.loadmat(self.dataFile)
        self.feature = self.data['data'][0]
        for i in range(self.feature.shape[0]):
            self.feature[i] = self.feature[i].T
        self.label = self.data['truelabel'][0][0].squeeze()
        self.idx = np.arange(self.feature[0].shape[0])
        self.graph_dict = {}
        self.mi_dict = {}
        err = []
        err_mi = []
        for i in range(self.feature.shape[0]):
            # print('Data size:', self.data['features'][0][i].shape)
            me = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                  'manhattan']
            g_all = []
            e_all = []
            mi_idx_all = []
            error_mi_all = []
            for met in me:
                g, e, mi_idx, error_mi = pair(self.mi, self.k, self.feature[i], self.label, metrix=met)
                g_all.append(g)
                e_all.append(e)
                mi_idx_all.append(mi_idx)
                error_mi_all.append(error_mi)
            # print('Graph pairs size:', g.shape)
            err.append(min(e_all))
            err_mi.append(min(error_mi_all))
            self.mi_dict[i] = mi_idx_all[error_mi_all.index(min(error_mi_all))]
            self.graph_dict[i] = g_all[e_all.index(min(e_all))]
        # print('Views:', self.graph_dict.keys())
        print('Views {}-nn errs:{}'.format(self.k, err))
        for i in range(len(err)):
            self._load(self.feature[i], self.label, self.idx, self.graph_dict[i], i)

    def _load(self, feature, label, idx, graph, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label)
        self.num_labels = labels.shape[1]

        # build graph
        idx = np.asarray(idx, dtype=np.int32)
        # print(idx)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = graph
        edges = np.asarray(list(map(idx_map.get, edges_unordered.flatten())),
                           dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph_dict[i] = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        features = _normalize(features)
        self.feature[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]

    def __len__(self):
        return 1


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot


def compare_1to1(x, y):
    temp = 1
    for i in range(x.shape[0]):
        if x[i] == 1 and y[i] == 1:
            temp += 1
    return temp
