from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from pygcn.utils import load_data, accuracy
# from pygcn.models import GCN
from utils import load_data, accuracy,evaluate_risk_preds
from models import GCN
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=201,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            # nclass=labels.max().item() + 1,
            nclass=2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# loss_op = torch.nn.CrossEntropyLoss(weight=torch.Tensor([10,1]).cuda())
loss_op = torch.nn.BCELoss(weight=torch.Tensor([100,1]).cuda())
# loss_op = torch.nn.CrossEntropyLoss()
# loss_op = torch.nn.MSELoss()
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # output = model(features, adj).view(-1)#这里如果要用二分类那种，如bceloss或MSEloss,model中最后返回1个列向量，需要转成行向量
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    a=output[idx_train]
    b=labels[idx_train]
    # loss_train = F.binary_cross_entropy(output[idx_train],labels[idx_train])
    loss_train = loss_op(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    split_loss, split_acc, split_recall, split_disturb, split_auc=evaluate_risk_preds(output, [labels], [idx_train], 0.3)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss_op(output[idx_val], labels[idx_val])
    c=output[idx_val]
    d=labels[idx_val]
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'split_loss:{:.4f}'.format(split_loss[0]),
          'split_acc:{:.4f}'.format(split_acc[0]),
          'split_recall:{:.4f}'.format(split_recall[0]),
          'split_disturb:{:.4f}'.format(split_disturb[0]),
          'split_auc:{:.4f}'.format(split_auc[0]),

          )


def test():
    model.eval()
    output = model(features, adj)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test])
    loss_test = loss_op(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


"""
1000轮
Epoch: 0942 loss_train: 1.3122 acc_train: 1.0000 loss_val: 9.4256 acc_val: 0.9300 time: 0.0163s split_loss:0.0099 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0943 loss_train: 1.2273 acc_train: 1.0000 loss_val: 9.5327 acc_val: 0.9300 time: 0.0266s split_loss:0.0085 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0944 loss_train: 1.1859 acc_train: 0.9933 loss_val: 9.6120 acc_val: 0.9300 time: 0.0199s split_loss:0.0084 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0945 loss_train: 1.0345 acc_train: 1.0000 loss_val: 9.6356 acc_val: 0.9300 time: 0.0197s split_loss:0.0080 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0946 loss_train: 1.1591 acc_train: 1.0000 loss_val: 9.6789 acc_val: 0.9300 time: 0.0201s split_loss:0.0080 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0947 loss_train: 1.0440 acc_train: 1.0000 loss_val: 9.6628 acc_val: 0.9300 time: 0.0200s split_loss:0.0079 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0948 loss_train: 1.1699 acc_train: 1.0000 loss_val: 9.6334 acc_val: 0.9300 time: 0.0209s split_loss:0.0088 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0949 loss_train: 1.1805 acc_train: 1.0000 loss_val: 9.6069 acc_val: 0.9300 time: 0.0192s split_loss:0.0088 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0950 loss_train: 1.2771 acc_train: 1.0000 loss_val: 9.6336 acc_val: 0.9300 time: 0.0194s split_loss:0.0084 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0951 loss_train: 1.0574 acc_train: 1.0000 loss_val: 9.6741 acc_val: 0.9300 time: 0.0193s split_loss:0.0076 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0952 loss_train: 1.8736 acc_train: 0.9933 loss_val: 9.6486 acc_val: 0.9300 time: 0.0194s split_loss:0.0147 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0953 loss_train: 1.6425 acc_train: 0.9933 loss_val: 9.6142 acc_val: 0.9300 time: 0.0197s split_loss:0.0129 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0954 loss_train: 1.3024 acc_train: 1.0000 loss_val: 9.5854 acc_val: 0.9300 time: 0.0193s split_loss:0.0097 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0955 loss_train: 1.2228 acc_train: 1.0000 loss_val: 9.5588 acc_val: 0.9300 time: 0.0195s split_loss:0.0093 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0956 loss_train: 1.3113 acc_train: 1.0000 loss_val: 9.5851 acc_val: 0.9300 time: 0.0196s split_loss:0.0091 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0957 loss_train: 1.0808 acc_train: 1.0000 loss_val: 9.6046 acc_val: 0.9300 time: 0.0195s split_loss:0.0079 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0958 loss_train: 1.6388 acc_train: 0.9933 loss_val: 9.5609 acc_val: 0.9300 time: 0.0194s split_loss:0.0136 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0959 loss_train: 0.8799 acc_train: 1.0000 loss_val: 9.4923 acc_val: 0.9300 time: 0.0194s split_loss:0.0067 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0960 loss_train: 1.0716 acc_train: 1.0000 loss_val: 9.4648 acc_val: 0.9300 time: 0.0198s split_loss:0.0073 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0961 loss_train: 1.4677 acc_train: 0.9933 loss_val: 9.4692 acc_val: 0.9300 time: 0.0197s split_loss:0.0110 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0962 loss_train: 1.3043 acc_train: 0.9933 loss_val: 9.5161 acc_val: 0.9300 time: 0.0188s split_loss:0.0095 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0963 loss_train: 1.3170 acc_train: 1.0000 loss_val: 9.5308 acc_val: 0.9300 time: 0.0179s split_loss:0.0106 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0964 loss_train: 1.0892 acc_train: 1.0000 loss_val: 9.5280 acc_val: 0.9300 time: 0.0173s split_loss:0.0085 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0965 loss_train: 1.4843 acc_train: 1.0000 loss_val: 9.5202 acc_val: 0.9300 time: 0.0172s split_loss:0.0110 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0966 loss_train: 1.1588 acc_train: 1.0000 loss_val: 9.5448 acc_val: 0.9300 time: 0.0169s split_loss:0.0085 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0967 loss_train: 1.0974 acc_train: 1.0000 loss_val: 9.5712 acc_val: 0.9300 time: 0.0169s split_loss:0.0083 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0968 loss_train: 0.9499 acc_train: 1.0000 loss_val: 9.6088 acc_val: 0.9300 time: 0.0166s split_loss:0.0063 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0969 loss_train: 1.2845 acc_train: 1.0000 loss_val: 9.6820 acc_val: 0.9300 time: 0.0165s split_loss:0.0093 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0970 loss_train: 1.4443 acc_train: 1.0000 loss_val: 9.6403 acc_val: 0.9300 time: 0.0168s split_loss:0.0118 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0971 loss_train: 1.1068 acc_train: 1.0000 loss_val: 9.5923 acc_val: 0.9300 time: 0.0164s split_loss:0.0077 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0972 loss_train: 1.3589 acc_train: 1.0000 loss_val: 9.4752 acc_val: 0.9300 time: 0.0164s split_loss:0.0112 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0973 loss_train: 1.1606 acc_train: 1.0000 loss_val: 9.4253 acc_val: 0.9300 time: 0.0167s split_loss:0.0080 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0974 loss_train: 1.4350 acc_train: 0.9933 loss_val: 9.4312 acc_val: 0.9300 time: 0.0192s split_loss:0.0101 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0975 loss_train: 1.1334 acc_train: 1.0000 loss_val: 9.4837 acc_val: 0.9300 time: 0.0194s split_loss:0.0071 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0976 loss_train: 1.3455 acc_train: 1.0000 loss_val: 9.5496 acc_val: 0.9300 time: 0.0200s split_loss:0.0099 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0977 loss_train: 1.1952 acc_train: 1.0000 loss_val: 9.6120 acc_val: 0.9300 time: 0.0194s split_loss:0.0087 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0978 loss_train: 1.2414 acc_train: 1.0000 loss_val: 9.6365 acc_val: 0.9300 time: 0.0196s split_loss:0.0103 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0979 loss_train: 1.3617 acc_train: 0.9933 loss_val: 9.5356 acc_val: 0.9300 time: 0.0189s split_loss:0.0106 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0980 loss_train: 1.3694 acc_train: 1.0000 loss_val: 9.4581 acc_val: 0.9300 time: 0.0215s split_loss:0.0103 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0981 loss_train: 0.9494 acc_train: 1.0000 loss_val: 9.4512 acc_val: 0.9300 time: 0.0188s split_loss:0.0063 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0982 loss_train: 1.2036 acc_train: 1.0000 loss_val: 9.4302 acc_val: 0.9300 time: 0.0215s split_loss:0.0095 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0983 loss_train: 1.2607 acc_train: 0.9933 loss_val: 9.4159 acc_val: 0.9300 time: 0.0218s split_loss:0.0100 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0984 loss_train: 0.8813 acc_train: 1.0000 loss_val: 9.4696 acc_val: 0.9300 time: 0.0213s split_loss:0.0056 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0985 loss_train: 1.4217 acc_train: 1.0000 loss_val: 9.5510 acc_val: 0.9300 time: 0.0188s split_loss:0.0105 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0986 loss_train: 1.1490 acc_train: 1.0000 loss_val: 9.6536 acc_val: 0.9300 time: 0.0178s split_loss:0.0078 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0987 loss_train: 0.8582 acc_train: 1.0000 loss_val: 9.7666 acc_val: 0.9300 time: 0.0178s split_loss:0.0059 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0988 loss_train: 1.5422 acc_train: 0.9867 loss_val: 9.8149 acc_val: 0.9300 time: 0.0181s split_loss:0.0114 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:0.9995
Epoch: 0989 loss_train: 1.2560 acc_train: 1.0000 loss_val: 9.7972 acc_val: 0.9300 time: 0.0183s split_loss:0.0095 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0990 loss_train: 1.0795 acc_train: 1.0000 loss_val: 9.7284 acc_val: 0.9300 time: 0.0186s split_loss:0.0084 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0991 loss_train: 1.2061 acc_train: 1.0000 loss_val: 9.6092 acc_val: 0.9300 time: 0.0187s split_loss:0.0094 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0992 loss_train: 1.3583 acc_train: 1.0000 loss_val: 9.4189 acc_val: 0.9300 time: 0.0233s split_loss:0.0110 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0993 loss_train: 1.3068 acc_train: 1.0000 loss_val: 9.2820 acc_val: 0.9300 time: 0.0234s split_loss:0.0101 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0994 loss_train: 1.5478 acc_train: 0.9933 loss_val: 9.2486 acc_val: 0.9300 time: 0.0237s split_loss:0.0109 split_acc:0.9867 split_recall:1.0000 split_disturb:0.0147 split_auc:0.9995
Epoch: 0995 loss_train: 1.3725 acc_train: 0.9933 loss_val: 9.2714 acc_val: 0.9300 time: 0.0233s split_loss:0.0101 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0996 loss_train: 1.6027 acc_train: 0.9933 loss_val: 9.3370 acc_val: 0.9300 time: 0.0234s split_loss:0.0116 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:0.9995
Epoch: 0997 loss_train: 1.2938 acc_train: 0.9933 loss_val: 9.4491 acc_val: 0.9300 time: 0.0235s split_loss:0.0093 split_acc:0.9933 split_recall:1.0000 split_disturb:0.0074 split_auc:1.0000
Epoch: 0998 loss_train: 1.0499 acc_train: 1.0000 loss_val: 9.5681 acc_val: 0.9300 time: 0.0234s split_loss:0.0076 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 0999 loss_train: 1.0005 acc_train: 1.0000 loss_val: 9.6519 acc_val: 0.9300 time: 0.0237s split_loss:0.0079 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Epoch: 1000 loss_train: 1.2995 acc_train: 1.0000 loss_val: 9.7478 acc_val: 0.9300 time: 0.0235s split_loss:0.0104 split_acc:1.0000 split_recall:1.0000 split_disturb:0.0000 split_auc:1.0000
Optimization Finished!
Total time elapsed: 23.7986s
Test set results: loss= 6.2794 accuracy= 0.9700"""