import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)#这里如果用MSEloss或者BCEloss要改成1，因为后面要过sigmoid
        # self.gc2 = GraphConvolution(nhid, 1)#这里如果用MSEloss或者BCEloss要改成1，因为后面要过sigmoid
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return F.sigmoid(x)
        # return x
