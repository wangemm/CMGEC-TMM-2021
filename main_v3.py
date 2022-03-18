import argparse
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from gae_v3 import GAE, GraphEmbedding, Dis
from dataset import *
from utils import *
from time import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy import linalg as LA
from sklearn.preprocessing import normalize
import csv

parser = argparse.ArgumentParser(description='GAE')
parser.add_argument('--k', type=int, default=7, help='number of k in k-nn')
parser.add_argument('--mi', type=int, default=3, help='number of k for mi')
parser.add_argument('--ptrain_epochs', '-pe', type=int, default=2000, help='number of pre-train_epochs')
parser.add_argument('--train_epochs', '-te', type=int, default=500, help='number of train_epochs')
parser.add_argument('--save_dir', '-s', type=str, default='./result', help='result directry')
parser.add_argument('--hidden_dimsV', type=int, nargs='+', default=[64, 32], help='list of V1 hidden dimensions')
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 10], help='list of feature hidden dimensions')
parser.add_argument('--heads', type=int, nargs='+', default=[4, 4], help='list of heads for GAT')
parser.add_argument('--plr', type=float, default=0.0001, help='Adam learning rate')
parser.add_argument('--tlr', type=float, default=0.01, help='Adam learning rate')
parser.add_argument('--lambda1', type=float, default=0.001, help='Rate for gtr')
parser.add_argument('--lambda2', type=float, default=1, help='Rate for clu')
parser.add_argument('--dataset', type=str, default='3sources', help='choose a dataset')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda:0" if args.cuda else "cpu")
print(args)


def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load and preprocess dataset
    data = load_data(args)
    views = len(data.feature)
    print('Number of Samples: {:02d}'.format(data.feature[0].shape[0]))
    print('Views:', views)

    feature0 = torch.FloatTensor(data.feature[0])
    feature1 = torch.FloatTensor(data.feature[1])
    feature2 = torch.FloatTensor(data.feature[2])

    print('VIEW-0 DateSize: {:02d} * {:02d} '.format(feature0.shape[0], feature0.shape[1]))
    print('VIEW-1 DateSize: {:02d} * {:02d} '.format(feature1.shape[0], feature1.shape[1]))
    print('VIEW-2 DateSize: {:02d} * {:02d} '.format(feature2.shape[0], feature2.shape[1]))

    in_feats = [feature0.shape[1], feature1.shape[1], feature2.shape[1]]

    graph0 = dgl.from_networkx(data.graph_dict[0])
    graph1 = dgl.from_networkx(data.graph_dict[1])
    graph2 = dgl.from_networkx(data.graph_dict[2])

    print('VIEW-0 Edges: {:02d} '.format(graph0.number_of_edges()))
    print('VIEW-1 Edges: {:02d} '.format(graph1.number_of_edges()))
    print('VIEW-2 Edges: {:02d} '.format(graph2.number_of_edges()))

    mik0 = data.mi_dict[0]
    mik1 = data.mi_dict[1]
    mik2 = data.mi_dict[2]
    mik = np.hstack((mik0, mik1, mik2))
    print('Mutual Information Matrix Size: ', mik.shape)

    degs0 = graph0.in_degrees().float()
    degs1 = graph1.in_degrees().float()
    degs2 = graph2.in_degrees().float()

    norm0 = torch.pow(degs0, -0.5)
    norm1 = torch.pow(degs1, -0.5)
    norm2 = torch.pow(degs2, -0.5)

    norm0[torch.isinf(norm0)] = 0
    norm1[torch.isinf(norm1)] = 0
    norm2[torch.isinf(norm2)] = 0

    graph0.ndata['norm'] = norm0.unsqueeze(1)
    graph1.ndata['norm'] = norm1.unsqueeze(1)
    graph2.ndata['norm'] = norm2.unsqueeze(1)

    adj0 = graph0.adjacency_matrix().to_dense()
    adj1 = graph1.adjacency_matrix().to_dense()
    adj2 = graph2.adjacency_matrix().to_dense()

    edges = graph0.number_of_edges() + graph1.number_of_edges() + graph2.number_of_edges()

    y = data.label
    n_clusters = len(np.unique(y))
    # print('Clusters:', n_clusters)

    # model
    model = GAE(in_feats, args.hidden_dimsV, args.hidden_dims, n_clusters, views)

    # print(model)
    model.train()
    model_g = GraphEmbedding(feature0.shape[0], int(feature0.shape[0] / 2), n_clusters)
    model_g.train()
    model_d = Dis(latent_dim=args.hidden_dims[-1])
    # optimizer
    optim_gae_p = torch.optim.Adam(model.parameters(), lr=args.plr)
    optim_gae_t = torch.optim.Adam(model.parameters(), lr=args.tlr)
    optim_ge_p = torch.optim.Adam(model_g.parameters(), lr=args.plr)
    optim_ge_t = torch.optim.Adam(model_g.parameters(), lr=args.tlr)
    # loss
    pos_weight = torch.Tensor([float(graph0.adjacency_matrix().to_dense().shape[0] ** 2 - edges / 2) / edges * 2])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_m = torch.nn.MSELoss()

    # To GPU
    criterion.cuda(device=device)
    criterion_m.cuda(device=device)
    model = model.to(device)
    model_g = model_g.to(device)
    model_d = model_d.to(device)

    graph0 = graph0.to(device)
    graph1 = graph1.to(device)
    graph2 = graph2.to(device)

    feature0 = feature0.to(device)
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)

    adj0 = adj0.to(device)
    adj1 = adj1.to(device)
    adj2 = adj2.to(device)

    # Pre-Train
    begin_time = time()
    print('GE Pre-Training Start')
    for epoch in range(args.ptrain_epochs):

        # train GraphEmbedding Network
        adjin = adj0
        adjin = torch.add(adjin, adj1)
        adjin = torch.add(adjin, adj2)

        adj_r = model_g.forward(adjin)
        loss_ge = (criterion_m(adj_r, adj0) + criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2)) / views
        optim_ge_p.zero_grad()
        loss_ge.backward()
        optim_ge_p.step()

        if (epoch + 1) % 200 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print(
                'GE-Pre-Training Epoch: {:02d} | GE-Loss: {:.5f} | Time: {:.2f}'.format(epoch + 1,
                                                                                        loss_ge,
                                                                                        run_time))

    # normalization
    adj_p = torch.clamp(adj_r, 0, 1)
    adj_p = torch.round(adj_p + 0.1)
    # build symmetric adjacency matrix
    adj_pn = adj_p.detach().cpu().numpy()
    adj_pn += adj_pn.T
    graph = nx.from_numpy_array(adj_pn, create_using=nx.DiGraph())
    graph = dgl.from_networkx(graph)
    graph = graph.to(device)

    print('GAE Pre-Training Start')
    for epoch in range(args.ptrain_epochs):
        # train GAE
        adj_logits, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)

        loss_rec = (criterion(adj_logits[0], adj0) + criterion(adj_logits[1], adj1) + criterion(adj_logits[2],
                                                                                                adj2)) / views
        optim_gae_p.zero_grad()
        loss_rec.backward()
        optim_gae_p.step()

        if (epoch + 1) % 200 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print('graph.number_of_edges', graph.number_of_edges())
            print(
                'GAE-Pre-Training Epoch: {:02d} | GAE-Loss: {:.5f} |  Time: {:.2f}'.format(epoch + 1,
                                                                                           loss_rec,
                                                                                           run_time))
    # obtain init clustering center
    with torch.no_grad():
        _, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    _, _, _, _, _ = eva(y, y_pred, 'Pre-train', args.dataset)

    # Training
    print('Training Start')
    for epoch in range(args.train_epochs):

        # train GFN
        adjin = adj0
        adjin = torch.add(adjin, adj1)
        adjin = torch.add(adjin, adj2)

        adj_r = model_g.forward(adjin)
        loss_gre = (criterion_m(adj_r, adj0) + criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2)) / views
        loss_gtr = trace_loss(adj_r, n_clusters) ** 2
        loss_ge = loss_gre + args.lambda1 * loss_gtr
        optim_ge_t.zero_grad()
        loss_ge.backward()
        optim_ge_t.step()

        # normalization
        adj_p = torch.clamp(adj_r, 0, 1)
        adj_p = torch.round(adj_p + 0.1)
        # build symmetric adjacency matrix
        adj_pn = adj_p.detach().cpu().numpy()
        adj_pn += adj_pn.T
        graph = nx.from_numpy_array(adj_pn)
        graph = dgl.from_networkx(graph)
        graph = graph.to(device)

        # Train GAE
        adj_logits, h = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
        loss_rec = (criterion(adj_logits[0], adj0) + criterion(adj_logits[1], adj1) + criterion(adj_logits[2],
                                                                                                adj2)) / views
        global_info_loss = 0
        for i in range(mik.shape[1]):
            h_shuffle = shuffling(h, latent=args.hidden_dims[-1])
            h_h_shuffle = torch.cat((h, h_shuffle), 1)
            h_h_shuffle_scores = model_d(h_h_shuffle)
            h_h = torch.cat((h, h[mik[:, i]]), 1)
            h_h_scores = model_d(h_h)
            global_info_loss += - torch.mean(torch.log(h_h_scores + 1e-6) + torch.log(1 - h_h_shuffle_scores + 1e-6))
        loss_gae = loss_rec + args.lambda2 * global_info_loss
        optim_gae_t.zero_grad()
        loss_gae.backward()
        optim_gae_t.step()

        # losses.append(loss.item())
        if (epoch + 1) % 200 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print(
                'Epoch: {:02d} | GAE-Loss: {:.5f} + {:.5f} =  {:.5f}| GE-Loss: {:.5f} + {:.5f} =  {:.5f} | Time: {:.2f}'.format(
                    epoch + 1, loss_rec, args.lambda2 * global_info_loss, loss_gae, loss_gre,
                    args.lambda1 * loss_gtr, loss_ge,
                    run_time))

    model.eval()
    _, z = model.forward(graph0, graph1, graph2, feature0, feature1, feature2, graph)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    acc, nmi, ari, ami, f1 = eva(y, y_pred, 'Final', args.dataset)
    return acc, nmi, ari, ami, f1


def trace_loss(adj, k):
    adj = torch.clamp(adj, 0, 1)
    adj = torch.round(adj)
    rowsum = adj.sum(axis=1).detach().cpu().numpy()
    d = torch.zeros(adj.shape).numpy()
    row, col = np.diag_indices_from(d)
    d[row, col] = rowsum
    l = d - adj.detach().cpu().numpy()
    e_vals, e_vecs = np.linalg.eig(l)
    sorted_indices = np.argsort(e_vals)
    q = torch.tensor(e_vecs[:, sorted_indices[0:k:]].astype(np.float32)).cuda()
    m = torch.mm(torch.t(q), adj)
    m = torch.mm(m, q)
    return torch.trace(m)


def normalized_cut(A, k):
    W = np.eye(len(A)) - normalize(A, norm='l1')
    eigvalues, eigvectors = LA.eig(W)
    indices = np.argsort(eigvalues)[1:k]
    return KMeans(n_clusters=k).fit_predict(np.abs(eigvectors[:, indices]))


def shuffling(x, latent):
    idxs = torch.arange(0, x.shape[0]).cuda()
    a = torch.randperm(idxs.size(0)).cuda()
    aa = idxs[a].unsqueeze(1)
    aaa = aa.repeat(1, latent)
    return torch.gather(x, 0, aaa)


if __name__ == '__main__':
    # for args.k in range(5,21):
    accA = []
    nmiA = []
    ariA = []
    amiA = []
    f1A = []
    for i in range(5):
        acc, nmi, ari, ami, f1 = main()
        accA.append(acc)
        nmiA.append(nmi)
        ariA.append(ari)
        amiA.append(ami)
        f1A.append(f1)

    print('ACC: ave|{:04f} std|{:04f}'.format(np.mean(accA), np.std(accA, ddof=1)))
    print('NMI: ave|{:04f} std|{:04f}'.format(np.mean(nmiA), np.std(nmiA, ddof=1)))
    print('ARI: ave|{:04f} std|{:04f}'.format(np.mean(ariA), np.std(ariA, ddof=1)))
    print('AMI: ave|{:04f} std|{:04f}'.format(np.mean(amiA), np.std(amiA, ddof=1)))
    print('F1:  ave|{:04f} std|{:04f}'.format(np.mean(f1A), np.std(f1A, ddof=1)))

