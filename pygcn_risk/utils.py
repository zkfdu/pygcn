import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error,roc_auc_score,confusion_matrix

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot




'''tjq处理数据代码
def load_risk_data(path="data/risk/", dataset="risk_day1_large_compo"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str), delimiter=',')
    features = idx_features_labels[:, 1:-1].astype(np.float16)
    labels = idx_features_labels[:, -1].astype(np.float16)

    # build graph
    idx = np.array(idx_features_labels[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels
def get_risk_splits(y):
    idx_train = range(150)
    idx_val = range(150, 250)
    idx_test = range(250, 450)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

'''

# def load_data(path="../data/cora/", dataset="cora"):
def load_data(path="/disk4/zk/charmsftp/ali_attention/pygcn/data/risk/", dataset="risk_day1_black_compo"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str),delimiter=',')
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1].astype(np.float16).astype(np.int32))
    # labels = idx_features_labels[:, -1].astype(np.float16).astype(np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(150)
    idx_val = range(150, 250)
    idx_test = range(250, 450)

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # labels = torch.LongTensor(labels)
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy_basic(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy(output, labels):#计算acc
    # output=F.sigmoid(output)
    
    # e=output.max(1)[0]
    # f=output.max(1)[1]
    g=output[:,1]>0.44
    # if output[:1]
    labels=labels.max(1)[1]
    # preds = output.max(1)[1].type_as(labels)
    preds = g.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def evaluate_risk_preds(preds, labels, indices, theta):
    '''
    train_val_loss, train_val_acc, train_val_recall, train_val_disturb = evaluate_risk_preds(preds, [y_train, y_val], [idx_train, idx_val], theta)
    '''

    preds=preds[:,1]
    # for i in labels:
    #     i=i.max(1)[1]
    # preds = g.type_as(labels)

    split_loss = list()
    split_auc = list()
    split_acc = list()
    split_recall = list()
    split_disturb = list()

    for y_split, idx_split in zip(labels, indices):
        y_split=y_split.max(1)[1]
        # y_true, y_pred = y_split[idx_split], preds[idx_split]
        y_true, y_pred = y_split[idx_split].flatten().cpu().detach().numpy(), preds[idx_split].flatten().cpu().detach().numpy()
        split_loss.append(mean_squared_error(y_true, y_pred))
        split_auc.append(roc_auc_score(y_true, y_pred))
        y_pred = np.where(y_pred >= theta, 1, 0)
        # print(y_true.shape, y_pred.shape, y_true.dtype, y_pred.dtype)
        # print(np.sum(y_pred), np.sum(y_true))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        split_acc.append((tn+tp)/(tn+tp+fp+fn))
        split_recall.append(tp/(tp+fn))
        split_disturb.append(fp/(tn+fp))


    return split_loss, split_acc, split_recall, split_disturb, split_auc