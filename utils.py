import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami_score
from sklearn.metrics.cluster import fowlkes_mallows_score as fmi_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro, precision_macro, recall_macro


def eva(y_true, y_pred, epoch=0, dataset='0', times=0):
    acc, f1, precision, recall = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = ami_score(y_true, y_pred, average_method='arithmetic')
    fmi = fmi_score(y_true, y_pred)
    '''
    if epoch == 'Final':
        np.savetxt("data/{:s}/{:d}.txt".format(dataset, int(acc * 10000)), y_pred.astype(int))
        f = open("data/{:s}/results.txt".format(dataset), mode='a')
        f.write(str(times) + str(epoch) + ':acc {:.5f}'.format(acc) + ', nmi {:.5f}'.format(nmi) + ', ari {:.5f}'.format(
            ari) + ', ami {:.5f}'.format(ami) + ', f1 {:.5f}'.format(f1))
        f.close()
    '''
    print(times, epoch, ':acc {:.5f}'.format(acc), ', nmi {:.5f}'.format(nmi), ', ari {:.5f}'.format(ari),
          ', ami {:.5f}'.format(ami),
          ', f1 {:.5f}'.format(f1))

    return acc, nmi, ari, ami, f1

def eva_only(y_true, y_pred, epoch=0, dataset='0', times=0):
    acc, f1, precision, recall = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = ami_score(y_true, y_pred, average_method='arithmetic')
    fmi = fmi_score(y_true, y_pred)
    '''
    if epoch == 'Final':
        np.savetxt("data/{:s}/{:d}.txt".format(dataset, int(acc * 10000)), y_pred.astype(int))
        f = open("data/{:s}/results.txt".format(dataset), mode='a')
        f.write(str(times) + str(epoch) + ':acc {:.5f}'.format(acc) + ', nmi {:.5f}'.format(nmi) + ', ari {:.5f}'.format(
            ari) + ', ami {:.5f}'.format(ami) + ', f1 {:.5f}'.format(f1))
        f.close()
    '''
    return acc, nmi, ari, ami,f1
