import dgl.function as fn
from dgl.nn.pytorch import GraphConv as GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims_v, hidden_dims, n_clusters, views, v=1):
        super(GAE, self).__init__()
        self.views = views

        # layer: v0
        if len(hidden_dims_v) >= 2:
            layers0 = [GCN(in_feats=in_dim[0], out_feats=hidden_dims_v[0], activation=F.relu)]
            for j in range(1, len(hidden_dims_v)):
                if j != len(hidden_dims_v) - 1:
                    layers0.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=F.relu))
                else:
                    layers0.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=lambda x: x))
        else:
            layers0 = [GCN(in_feats=in_dim[0], out_feats=hidden_dims_v[0], activation=lambda x: x)]

        # layer: v1
        if len(hidden_dims_v) >= 2:
            layers1 = [GCN(in_feats=in_dim[1], out_feats=hidden_dims_v[0], activation=F.relu)]
            for j in range(1, len(hidden_dims_v)):
                if j != len(hidden_dims_v) - 1:
                    layers1.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=F.relu))
                else:
                    layers1.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=lambda x: x))
        else:
            layers1 = [GCN(in_feats=in_dim[1], out_feats=hidden_dims_v[0], activation=lambda x: x)]

        # layer: v2
        if len(hidden_dims_v) >= 2:
            layers2 = [GCN(in_feats=in_dim[2], out_feats=hidden_dims_v[0], activation=F.relu)]
            for j in range(1, len(hidden_dims_v)):
                if j != len(hidden_dims_v) - 1:
                    layers2.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=F.relu))
                else:
                    layers2.append(
                        GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=lambda x: x))
        else:
            layers2 = [GCN(in_feats=in_dim[2], out_feats=hidden_dims_v[0], activation=lambda x: x)]

        # layer: vm
        if len(hidden_dims) >= 2:
            layer_m = [GCN(in_feats=int(hidden_dims_v[-1]), out_feats=hidden_dims[0], activation=F.relu)]
            for i in range(1, len(hidden_dims)):
                if i != len(hidden_dims) - 1:
                    layer_m.append(GCN(in_feats=hidden_dims[i - 1], out_feats=hidden_dims[i], activation=F.relu))
                else:
                    layer_m.append(GCN(in_feats=hidden_dims[i - 1], out_feats=hidden_dims[i], activation=lambda x: x))
        else:
            layer_m = [GCN(in_feats=int(hidden_dims_v[-1] * views), out_feats=hidden_dims[0], activation=lambda x: x)]

        self.layer = nn.ModuleList(layer_m)
        self.layer0 = nn.ModuleList(layers0)
        self.layer1 = nn.ModuleList(layers1)
        self.layer2 = nn.ModuleList(layers2)
        self.featfusion = FeatureFusion(size=hidden_dims_v[-1])
        self.decoder = InnerProductDecoder(activation=lambda x: x, size=hidden_dims[-1])
        self.v = v

    def forward(self, graph0, graph1, graph2, feature0, feature1, feature2, graph):
        h0 = feature0
        h1 = feature1
        h2 = feature2

        for conv in self.layer0:
            h0 = conv(graph0, h0)
        for conv in self.layer1:
            h1 = conv(graph1, h1)
        for conv in self.layer2:
            h2 = conv(graph2, h2)
        h = torch.cat([h0, h1, h2], 1)
        xh = self.featfusion(h0, h1, h2)
        for conv in self.layer:
            if conv == self.layer[0]:
                xh = conv(dgl.add_self_loop(graph), xh)
            else:
                xh = conv(dgl.add_self_loop(graph), xh)
        adj_rec = {}
        for i in range(self.views):
            adj_rec[i] = self.decoder(xh)

        return adj_rec, xh


class FeatureFusion(nn.Module):
    def __init__(self, activation=torch.relu, dropout=0.1, size=64):
        super(FeatureFusion, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weight0 = Parameter(torch.FloatTensor(size, size))
        self.weight1 = Parameter(torch.FloatTensor(size, size))
        self.weight2 = Parameter(torch.FloatTensor(size, size))
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, z1, z2, z3):
        z1 = F.dropout(z1, self.dropout)
        z2 = F.dropout(z2, self.dropout)
        z3 = F.dropout(z3, self.dropout)
        t = torch.mm(z1, self.weight0) + torch.mm(z2, self.weight1) + torch.mm(z3, self.weight2)
        t = F.softmax(t, dim=1)
        t = self.activation(t)
        return t


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1, size=10):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(size, size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        t = torch.mm(z, self.weight)
        adj = self.activation(torch.mm(z, z.t()))
        return adj


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, n_clusters):
        super(GraphEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, n_clusters)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        predict = F.softmax(out, dim=1)
        return predict


class Dis(nn.Module):
    def __init__(self, latent_dim=120):
        super(Dis, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.discriminator(z)
        return out