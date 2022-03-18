import os
import numpy as np
from random import randint
from sklearn.neighbors import NearestNeighbors
import scipy.io as scio


def pair(mik=3, knn=10, data='', label='', metrix='minkowski'):
    '''
    dataFile = '../MV_datasets/processing/Mfeat/mfeat.mat'
    data = scio.loadmat(dataFile)
    feature = data['features'][0]
    feature[0] = feature[0].squeeze()
    label = data['labels'].T.squeeze()
    # idx = np.arange(feature[0].shape[0])
    '''
    # print(np.isnan(data))
    # print(np.isfinite(data).all())
    # print(np.isinf(data).all())
    # print(np.isnan(label).any())
    # print(np.isfinite(label))
    # print(np.isinf(label).all())
    # x = np.argwhere(np.isnan(data))
    # print(x[:,0])
    x_train, y_train = data, label

    # print('x_train shape', x_train.shape)
    # print('y_train shape', y_train.shape)

    # KNN group
    n_train = len(x_train)
    # print('computing k={} nearest neighbors...'.format(knn))

    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    # print('x_train_flat', x_train_flat.shape)

    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)

    # print('idx shape before', idx.shape)

    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()
    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e
    idx = new_idx.astype(np.int)
    mi_idx = idx[:, :mik]
    k_max = min(idx.shape[1], knn + 1)

    # print('idx shape after', idx.shape)
    # print('idx', idx)

    counter = 0
    counter_mi = 0

    for i, v in enumerate(idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter += 1
    error = counter / (y_train.shape[0] * knn)
    # print('error rate: {}'.format(error))
    for i, v in enumerate(mi_idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter_mi += 1
    error_mi = counter_mi / (y_train.shape[0] * mik)
    graph = np.empty(shape=[0, 2], dtype=int)
    for i, m in enumerate(idx):
        for mm in m:
            # print(i, mm)
            graph = np.append(graph, [[i, mm]], axis=0)

    # print(graph.shape)
    # print(graph)
    return graph, error, mi_idx, error_mi


def get_choices(arr, num_choices, valid_range=[-1, np.inf], not_arr=None, replace=False):
    '''
    Select n=num_choices choices from arr, with the following constraints for
    each choice:
        choice > valid_range[0],
        choice < valid_range[1],
        choice not in not_arr
    if replace == True, draw choices with replacement
    if arr is an integer, the pool of choices is interpreted as [0, arr]
    (inclusive)
        * in the implementation, we use an identity function to create the
        identity map arr[i] = i
    '''

    if not_arr is None:
        not_arr = []
    if isinstance(valid_range, int):
        valid_range = [0, valid_range]
    # make sure we have enough valid points in arr
    if isinstance(arr, tuple):
        if min(arr[1], valid_range[1]) - max(arr[0], valid_range[0]) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = arr[1]
        arr0 = arr[0]
        arr = defaultdict(lambda: -1)
        get_arr = lambda x: x
        replace = True
    else:
        greater_than = np.array(arr) > valid_range[0]
        less_than = np.array(arr) < valid_range[1]
        if np.sum(np.logical_and(greater_than, less_than)) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        # make a copy of arr, since we'll be editing the array
        n_arr = len(arr)
        arr0 = 0
        arr = np.array(arr, copy=True)
        get_arr = lambda x: arr[x]
    not_arr_set = set(not_arr)

    def get_choice():
        arr_idx = randint(arr0, n_arr - 1)
        while get_arr(arr_idx) in not_arr_set:
            arr_idx = randint(arr0, n_arr - 1)
        return arr_idx

    if isinstance(not_arr, int):
        not_arr = list(not_arr)
    choices = []
    for _ in range(num_choices):
        arr_idx = get_choice()
        while get_arr(arr_idx) <= valid_range[0] or get_arr(arr_idx) >= valid_range[1]:
            arr_idx = get_choice()
        choices.append(int(get_arr(arr_idx)))
        if not replace:
            arr[arr_idx], arr[n_arr - 1] = arr[n_arr - 1], arr[arr_idx]
            n_arr -= 1
    return choices


'''
if __name__ == '__main__':
    g, e = pair()
'''
