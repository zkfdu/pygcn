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
parser.add_argument('--epochs', type=int, default=200,
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
loss_op = torch.nn.BCELoss(weight=torch.Tensor([10,1]).cuda())
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
(aliatte)  ✘ zk@E211  /disk4/zk/charmsftp/ali_attention  cd /disk4/zk/charmsftp/ali_attention ; env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1 /disk2/zk/sw/Anaconda2/envs/aliatte/bin/python /disk2/zk/.vscode-server-insiders/extensions/ms-python.python-2019.6.24221/pythonFiles/ptvsd_launcher.py --default --client --host localhost --port 36241 /disk4/zk/charmsftp/ali_attention/pygcn/pygcn_risk/train.py 
Loading risk_day1_black_compo dataset...
Epoch: 0001 loss_train: 3.1386 acc_train: 0.8933 loss_val: 3.0380 acc_val: 0.9000 time: 0.6778s split_loss:0.1915 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.4611
Epoch: 0002 loss_train: 3.0280 acc_train: 0.9067 loss_val: 2.9195 acc_val: 0.9000 time: 0.0154s split_loss:0.1844 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.3151
Epoch: 0003 loss_train: 2.9086 acc_train: 0.9000 loss_val: 2.8044 acc_val: 0.9000 time: 0.0152s split_loss:0.1765 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.6780
Epoch: 0004 loss_train: 2.7900 acc_train: 0.9067 loss_val: 2.6984 acc_val: 0.9000 time: 0.0144s split_loss:0.1696 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.6544
Epoch: 0005 loss_train: 2.6705 acc_train: 0.9067 loss_val: 2.5998 acc_val: 0.9000 time: 0.0149s split_loss:0.1646 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.6087
Epoch: 0006 loss_train: 2.5795 acc_train: 0.9067 loss_val: 2.5073 acc_val: 0.9000 time: 0.0159s split_loss:0.1595 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.4501
Epoch: 0007 loss_train: 2.5080 acc_train: 0.9067 loss_val: 2.4212 acc_val: 0.9000 time: 0.0154s split_loss:0.1545 split_acc:0.0933 split_recall:1.0000 split_disturb:1.0000 split_auc:0.7001
Epoch: 0008 loss_train: 2.3883 acc_train: 0.9067 loss_val: 2.3415 acc_val: 0.9000 time: 0.0147s split_loss:0.1486 split_acc:0.1000 split_recall:1.0000 split_disturb:0.9926 split_auc:0.4706
Epoch: 0009 loss_train: 2.3207 acc_train: 0.9067 loss_val: 2.2673 acc_val: 0.9000 time: 0.0169s split_loss:0.1454 split_acc:0.1067 split_recall:1.0000 split_disturb:0.9853 split_auc:0.5651
Epoch: 0010 loss_train: 2.2365 acc_train: 0.9067 loss_val: 2.1980 acc_val: 0.9000 time: 0.0163s split_loss:0.1404 split_acc:0.1067 split_recall:0.7857 split_disturb:0.9632 split_auc:0.4984
Epoch: 0011 loss_train: 2.1657 acc_train: 0.9067 loss_val: 2.1339 acc_val: 0.9000 time: 0.0155s split_loss:0.1379 split_acc:0.1467 split_recall:0.7857 split_disturb:0.9191 split_auc:0.4895
Epoch: 0012 loss_train: 2.1366 acc_train: 0.9067 loss_val: 2.0752 acc_val: 0.9000 time: 0.0165s split_loss:0.1346 split_acc:0.3067 split_recall:0.2857 split_disturb:0.6912 split_auc:0.2889
Epoch: 0013 loss_train: 2.0300 acc_train: 0.9067 loss_val: 2.0219 acc_val: 0.9000 time: 0.0205s split_loss:0.1267 split_acc:0.5800 split_recall:0.5714 split_disturb:0.4191 split_auc:0.5819
Epoch: 0014 loss_train: 1.9869 acc_train: 0.9067 loss_val: 1.9742 acc_val: 0.9000 time: 0.0201s split_loss:0.1235 split_acc:0.6200 split_recall:0.5714 split_disturb:0.3750 split_auc:0.5504
Epoch: 0015 loss_train: 1.9593 acc_train: 0.9067 loss_val: 1.9322 acc_val: 0.9000 time: 0.0202s split_loss:0.1214 split_acc:0.7733 split_recall:0.2143 split_disturb:0.1691 split_auc:0.4112
Epoch: 0016 loss_train: 1.8999 acc_train: 0.9067 loss_val: 1.8958 acc_val: 0.9000 time: 0.0202s split_loss:0.1167 split_acc:0.8133 split_recall:0.2857 split_disturb:0.1324 split_auc:0.5021
Epoch: 0017 loss_train: 1.8740 acc_train: 0.9067 loss_val: 1.8648 acc_val: 0.9000 time: 0.0190s split_loss:0.1118 split_acc:0.8667 split_recall:0.2857 split_disturb:0.0735 split_auc:0.5930
Epoch: 0018 loss_train: 1.8403 acc_train: 0.9067 loss_val: 1.8390 acc_val: 0.9000 time: 0.0153s split_loss:0.1101 split_acc:0.8333 split_recall:0.2143 split_disturb:0.1029 split_auc:0.5310
Epoch: 0019 loss_train: 1.8598 acc_train: 0.9067 loss_val: 1.8181 acc_val: 0.9000 time: 0.0153s split_loss:0.1109 split_acc:0.8800 split_recall:0.0714 split_disturb:0.0368 split_auc:0.3209
Epoch: 0020 loss_train: 1.7671 acc_train: 0.9067 loss_val: 1.8016 acc_val: 0.9000 time: 0.0160s split_loss:0.1033 split_acc:0.8800 split_recall:0.0000 split_disturb:0.0294 split_auc:0.6224
Epoch: 0021 loss_train: 1.7623 acc_train: 0.9067 loss_val: 1.7891 acc_val: 0.9000 time: 0.0151s split_loss:0.1006 split_acc:0.8933 split_recall:0.0714 split_disturb:0.0221 split_auc:0.4921
Epoch: 0022 loss_train: 1.7610 acc_train: 0.9067 loss_val: 1.7801 acc_val: 0.9000 time: 0.0154s split_loss:0.1017 split_acc:0.8800 split_recall:0.0000 split_disturb:0.0294 split_auc:0.4984
Epoch: 0023 loss_train: 1.7083 acc_train: 0.9067 loss_val: 1.7741 acc_val: 0.9000 time: 0.0157s split_loss:0.0978 split_acc:0.9000 split_recall:0.0714 split_disturb:0.0147 split_auc:0.5000
Epoch: 0024 loss_train: 1.6875 acc_train: 0.9067 loss_val: 1.7706 acc_val: 0.9000 time: 0.0146s split_loss:0.0938 split_acc:0.9000 split_recall:0.0000 split_disturb:0.0074 split_auc:0.6176
Epoch: 0025 loss_train: 1.6793 acc_train: 0.9067 loss_val: 1.7692 acc_val: 0.9000 time: 0.0139s split_loss:0.0931 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5315
Epoch: 0026 loss_train: 1.8114 acc_train: 0.9067 loss_val: 1.7692 acc_val: 0.9000 time: 0.0144s split_loss:0.0961 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.3713
Epoch: 0027 loss_train: 1.7350 acc_train: 0.9067 loss_val: 1.7701 acc_val: 0.9000 time: 0.0139s split_loss:0.0887 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.5546
Epoch: 0028 loss_train: 1.7836 acc_train: 0.9067 loss_val: 1.7715 acc_val: 0.9000 time: 0.0141s split_loss:0.0891 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5100
Epoch: 0029 loss_train: 1.7884 acc_train: 0.9067 loss_val: 1.7729 acc_val: 0.9000 time: 0.0145s split_loss:0.0892 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5362
Epoch: 0030 loss_train: 1.7123 acc_train: 0.9067 loss_val: 1.7743 acc_val: 0.9000 time: 0.0146s split_loss:0.0901 split_acc:0.9000 split_recall:0.0714 split_disturb:0.0147 split_auc:0.4422
Epoch: 0031 loss_train: 1.6090 acc_train: 0.9067 loss_val: 1.7755 acc_val: 0.9000 time: 0.0145s split_loss:0.0826 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.6308
Epoch: 0032 loss_train: 1.6778 acc_train: 0.9067 loss_val: 1.7763 acc_val: 0.9000 time: 0.0143s split_loss:0.0836 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6371
Epoch: 0033 loss_train: 1.7128 acc_train: 0.9067 loss_val: 1.7765 acc_val: 0.9000 time: 0.0146s split_loss:0.0855 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5646
Epoch: 0034 loss_train: 1.8413 acc_train: 0.9067 loss_val: 1.7758 acc_val: 0.9000 time: 0.0146s split_loss:0.0878 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.4995
Epoch: 0035 loss_train: 1.6331 acc_train: 0.9067 loss_val: 1.7745 acc_val: 0.9000 time: 0.0146s split_loss:0.0826 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.6061
Epoch: 0036 loss_train: 1.6329 acc_train: 0.9067 loss_val: 1.7727 acc_val: 0.9000 time: 0.0146s split_loss:0.0807 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6597
Epoch: 0037 loss_train: 1.7514 acc_train: 0.9067 loss_val: 1.7702 acc_val: 0.9000 time: 0.0145s split_loss:0.0872 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.4653
Epoch: 0038 loss_train: 1.6840 acc_train: 0.9067 loss_val: 1.7673 acc_val: 0.9000 time: 0.0146s split_loss:0.0830 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.5562
Epoch: 0039 loss_train: 1.6073 acc_train: 0.9067 loss_val: 1.7644 acc_val: 0.9000 time: 0.0146s split_loss:0.0823 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6234
Epoch: 0040 loss_train: 1.7753 acc_train: 0.9067 loss_val: 1.7609 acc_val: 0.9000 time: 0.0155s split_loss:0.0865 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.4669
Epoch: 0041 loss_train: 1.7455 acc_train: 0.9067 loss_val: 1.7572 acc_val: 0.9000 time: 0.0187s split_loss:0.0859 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5068
Epoch: 0042 loss_train: 1.7273 acc_train: 0.9067 loss_val: 1.7533 acc_val: 0.9000 time: 0.0189s split_loss:0.0839 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5441
Epoch: 0043 loss_train: 1.6965 acc_train: 0.9067 loss_val: 1.7494 acc_val: 0.9000 time: 0.0188s split_loss:0.0831 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5877
Epoch: 0044 loss_train: 1.6519 acc_train: 0.9067 loss_val: 1.7456 acc_val: 0.9000 time: 0.0148s split_loss:0.0823 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6397
Epoch: 0045 loss_train: 1.5414 acc_train: 0.9067 loss_val: 1.7424 acc_val: 0.9000 time: 0.0178s split_loss:0.0795 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6880
Epoch: 0046 loss_train: 1.5743 acc_train: 0.9067 loss_val: 1.7394 acc_val: 0.9000 time: 0.0181s split_loss:0.0820 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6066
Epoch: 0047 loss_train: 1.5746 acc_train: 0.9067 loss_val: 1.7368 acc_val: 0.9000 time: 0.0184s split_loss:0.0792 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7878
Epoch: 0048 loss_train: 1.7318 acc_train: 0.9067 loss_val: 1.7344 acc_val: 0.9000 time: 0.0186s split_loss:0.0857 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5089
Epoch: 0049 loss_train: 1.6832 acc_train: 0.9067 loss_val: 1.7323 acc_val: 0.9000 time: 0.0185s split_loss:0.0833 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5326
Epoch: 0050 loss_train: 1.7057 acc_train: 0.9067 loss_val: 1.7307 acc_val: 0.9000 time: 0.0185s split_loss:0.0822 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6591
Epoch: 0051 loss_train: 1.6307 acc_train: 0.9067 loss_val: 1.7291 acc_val: 0.9000 time: 0.0184s split_loss:0.0835 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6318
Epoch: 0052 loss_train: 1.6679 acc_train: 0.9067 loss_val: 1.7278 acc_val: 0.9000 time: 0.0166s split_loss:0.0823 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6502
Epoch: 0053 loss_train: 1.6811 acc_train: 0.9067 loss_val: 1.7265 acc_val: 0.9000 time: 0.0148s split_loss:0.0846 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5368
Epoch: 0054 loss_train: 1.6933 acc_train: 0.9067 loss_val: 1.7254 acc_val: 0.9000 time: 0.0148s split_loss:0.0834 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5930
Epoch: 0055 loss_train: 1.6265 acc_train: 0.9067 loss_val: 1.7242 acc_val: 0.9000 time: 0.0147s split_loss:0.0822 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6576
Epoch: 0056 loss_train: 1.6976 acc_train: 0.9067 loss_val: 1.7230 acc_val: 0.9000 time: 0.0145s split_loss:0.0852 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.4538
Epoch: 0057 loss_train: 1.6539 acc_train: 0.9067 loss_val: 1.7219 acc_val: 0.9000 time: 0.0146s split_loss:0.0826 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6597
Epoch: 0058 loss_train: 1.7062 acc_train: 0.9067 loss_val: 1.7208 acc_val: 0.9000 time: 0.0147s split_loss:0.0843 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5646
Epoch: 0059 loss_train: 1.6407 acc_train: 0.9067 loss_val: 1.7196 acc_val: 0.9000 time: 0.0148s split_loss:0.0828 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5530
Epoch: 0060 loss_train: 1.5257 acc_train: 0.9067 loss_val: 1.7184 acc_val: 0.9000 time: 0.0147s split_loss:0.0782 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7038
Epoch: 0061 loss_train: 1.5868 acc_train: 0.9067 loss_val: 1.7172 acc_val: 0.9000 time: 0.0149s split_loss:0.0805 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6985
Epoch: 0062 loss_train: 1.6117 acc_train: 0.9067 loss_val: 1.7159 acc_val: 0.9000 time: 0.0147s split_loss:0.0836 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6371
Epoch: 0063 loss_train: 1.6514 acc_train: 0.9067 loss_val: 1.7146 acc_val: 0.9000 time: 0.0145s split_loss:0.0835 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5762
Epoch: 0064 loss_train: 1.5671 acc_train: 0.9067 loss_val: 1.7133 acc_val: 0.9000 time: 0.0146s split_loss:0.0808 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6954
Epoch: 0065 loss_train: 1.6283 acc_train: 0.9067 loss_val: 1.7120 acc_val: 0.9000 time: 0.0146s split_loss:0.0836 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6145
Epoch: 0066 loss_train: 1.6307 acc_train: 0.9067 loss_val: 1.7106 acc_val: 0.9000 time: 0.0148s split_loss:0.0837 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5924
Epoch: 0067 loss_train: 1.5708 acc_train: 0.9067 loss_val: 1.7093 acc_val: 0.9000 time: 0.0146s split_loss:0.0804 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7731
Epoch: 0068 loss_train: 1.6143 acc_train: 0.9067 loss_val: 1.7080 acc_val: 0.9000 time: 0.0145s split_loss:0.0816 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6113
Epoch: 0069 loss_train: 1.5977 acc_train: 0.9067 loss_val: 1.7066 acc_val: 0.9000 time: 0.0144s split_loss:0.0812 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6707
Epoch: 0070 loss_train: 1.6638 acc_train: 0.9067 loss_val: 1.7052 acc_val: 0.9000 time: 0.0145s split_loss:0.0829 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6050
Epoch: 0071 loss_train: 1.6126 acc_train: 0.9067 loss_val: 1.7037 acc_val: 0.9000 time: 0.0145s split_loss:0.0815 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6565
Epoch: 0072 loss_train: 1.5571 acc_train: 0.9067 loss_val: 1.7023 acc_val: 0.9000 time: 0.0145s split_loss:0.0799 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8078
Epoch: 0073 loss_train: 1.5441 acc_train: 0.9067 loss_val: 1.7010 acc_val: 0.9000 time: 0.0146s split_loss:0.0799 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7878
Epoch: 0074 loss_train: 1.6297 acc_train: 0.9067 loss_val: 1.6996 acc_val: 0.9000 time: 0.0145s split_loss:0.0837 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5646
Epoch: 0075 loss_train: 1.6121 acc_train: 0.9067 loss_val: 1.6980 acc_val: 0.9000 time: 0.0147s split_loss:0.0818 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6155
Epoch: 0076 loss_train: 1.5931 acc_train: 0.9067 loss_val: 1.6965 acc_val: 0.9000 time: 0.0144s split_loss:0.0819 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6345
Epoch: 0077 loss_train: 1.6304 acc_train: 0.9067 loss_val: 1.6949 acc_val: 0.9000 time: 0.0145s split_loss:0.0826 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6481
Epoch: 0078 loss_train: 1.5828 acc_train: 0.9067 loss_val: 1.6931 acc_val: 0.9000 time: 0.0145s split_loss:0.0816 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7117
Epoch: 0079 loss_train: 1.6465 acc_train: 0.9067 loss_val: 1.6911 acc_val: 0.9000 time: 0.0145s split_loss:0.0838 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.5467
Epoch: 0080 loss_train: 1.5194 acc_train: 0.9067 loss_val: 1.6891 acc_val: 0.9000 time: 0.0146s split_loss:0.0785 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7243
Epoch: 0081 loss_train: 1.5254 acc_train: 0.9067 loss_val: 1.6871 acc_val: 0.9000 time: 0.0145s split_loss:0.0809 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7222
Epoch: 0082 loss_train: 1.4928 acc_train: 0.9067 loss_val: 1.6851 acc_val: 0.9000 time: 0.0144s split_loss:0.0788 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8162
Epoch: 0083 loss_train: 1.5491 acc_train: 0.9067 loss_val: 1.6830 acc_val: 0.9000 time: 0.0145s split_loss:0.0813 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7164
Epoch: 0084 loss_train: 1.5600 acc_train: 0.9067 loss_val: 1.6809 acc_val: 0.9000 time: 0.0143s split_loss:0.0820 split_acc:0.9000 split_recall:0.0000 split_disturb:0.0074 split_auc:0.6628
Epoch: 0085 loss_train: 1.5757 acc_train: 0.9067 loss_val: 1.6789 acc_val: 0.9000 time: 0.0145s split_loss:0.0814 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7227
Epoch: 0086 loss_train: 1.5254 acc_train: 0.9067 loss_val: 1.6768 acc_val: 0.9000 time: 0.0153s split_loss:0.0816 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6970
Epoch: 0087 loss_train: 1.5946 acc_train: 0.9067 loss_val: 1.6748 acc_val: 0.9000 time: 0.0178s split_loss:0.0815 split_acc:0.9000 split_recall:0.0000 split_disturb:0.0074 split_auc:0.7038
Epoch: 0088 loss_train: 1.4855 acc_train: 0.9067 loss_val: 1.6727 acc_val: 0.9000 time: 0.0184s split_loss:0.0787 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7747
Epoch: 0089 loss_train: 1.5921 acc_train: 0.9067 loss_val: 1.6708 acc_val: 0.9000 time: 0.0164s split_loss:0.0828 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6896
Epoch: 0090 loss_train: 1.5585 acc_train: 0.9067 loss_val: 1.6688 acc_val: 0.9000 time: 0.0153s split_loss:0.0810 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6991
Epoch: 0091 loss_train: 1.4856 acc_train: 0.9067 loss_val: 1.6667 acc_val: 0.9000 time: 0.0153s split_loss:0.0785 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8046
Epoch: 0092 loss_train: 1.4677 acc_train: 0.9067 loss_val: 1.6646 acc_val: 0.9000 time: 0.0152s split_loss:0.0782 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.7789
Epoch: 0093 loss_train: 1.4178 acc_train: 0.9067 loss_val: 1.6623 acc_val: 0.9000 time: 0.0150s split_loss:0.0746 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.9312
Epoch: 0094 loss_train: 1.5689 acc_train: 0.9067 loss_val: 1.6600 acc_val: 0.9000 time: 0.0151s split_loss:0.0801 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.6938
Epoch: 0095 loss_train: 1.4424 acc_train: 0.9067 loss_val: 1.6576 acc_val: 0.9000 time: 0.0151s split_loss:0.0781 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8204
Epoch: 0096 loss_train: 1.4759 acc_train: 0.9067 loss_val: 1.6554 acc_val: 0.9000 time: 0.0152s split_loss:0.0777 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8020
Epoch: 0097 loss_train: 1.5612 acc_train: 0.9067 loss_val: 1.6529 acc_val: 0.9000 time: 0.0174s split_loss:0.0813 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6922
Epoch: 0098 loss_train: 1.5064 acc_train: 0.9067 loss_val: 1.6505 acc_val: 0.9000 time: 0.0269s split_loss:0.0813 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7096
Epoch: 0099 loss_train: 1.5125 acc_train: 0.9067 loss_val: 1.6482 acc_val: 0.9000 time: 0.0187s split_loss:0.0790 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7458
Epoch: 0100 loss_train: 1.4505 acc_train: 0.9067 loss_val: 1.6456 acc_val: 0.9000 time: 0.0185s split_loss:0.0786 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7952
Epoch: 0101 loss_train: 1.5288 acc_train: 0.9067 loss_val: 1.6430 acc_val: 0.9000 time: 0.0186s split_loss:0.0792 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.6418
Epoch: 0102 loss_train: 1.3871 acc_train: 0.9067 loss_val: 1.6403 acc_val: 0.9000 time: 0.0188s split_loss:0.0756 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.9375
Epoch: 0103 loss_train: 1.4896 acc_train: 0.9067 loss_val: 1.6376 acc_val: 0.9000 time: 0.0181s split_loss:0.0787 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7348
Epoch: 0104 loss_train: 1.4421 acc_train: 0.9067 loss_val: 1.6349 acc_val: 0.9000 time: 0.0174s split_loss:0.0760 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8067
Epoch: 0105 loss_train: 1.4052 acc_train: 0.9067 loss_val: 1.6322 acc_val: 0.9000 time: 0.0146s split_loss:0.0755 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8514
Epoch: 0106 loss_train: 1.4078 acc_train: 0.9067 loss_val: 1.6294 acc_val: 0.9000 time: 0.0147s split_loss:0.0767 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7936
Epoch: 0107 loss_train: 1.4667 acc_train: 0.9067 loss_val: 1.6268 acc_val: 0.9000 time: 0.0147s split_loss:0.0774 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8267
Epoch: 0108 loss_train: 1.4885 acc_train: 0.9067 loss_val: 1.6242 acc_val: 0.9000 time: 0.0147s split_loss:0.0785 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7736
Epoch: 0109 loss_train: 1.5093 acc_train: 0.9067 loss_val: 1.6217 acc_val: 0.9000 time: 0.0144s split_loss:0.0775 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7794
Epoch: 0110 loss_train: 1.4297 acc_train: 0.9067 loss_val: 1.6192 acc_val: 0.9000 time: 0.0192s split_loss:0.0749 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8057
Epoch: 0111 loss_train: 1.4762 acc_train: 0.9067 loss_val: 1.6169 acc_val: 0.9000 time: 0.0287s split_loss:0.0785 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.7353
Epoch: 0112 loss_train: 1.3969 acc_train: 0.9067 loss_val: 1.6145 acc_val: 0.9000 time: 0.0150s split_loss:0.0759 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8545
Epoch: 0113 loss_train: 1.3595 acc_train: 0.9067 loss_val: 1.6121 acc_val: 0.9000 time: 0.0147s split_loss:0.0744 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8955
Epoch: 0114 loss_train: 1.3660 acc_train: 0.9067 loss_val: 1.6102 acc_val: 0.9000 time: 0.0146s split_loss:0.0737 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.9380
Epoch: 0115 loss_train: 1.4479 acc_train: 0.9067 loss_val: 1.6080 acc_val: 0.9000 time: 0.0144s split_loss:0.0773 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8062
Epoch: 0116 loss_train: 1.4092 acc_train: 0.9067 loss_val: 1.6062 acc_val: 0.9000 time: 0.0144s split_loss:0.0760 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.8209
Epoch: 0117 loss_train: 1.3991 acc_train: 0.9067 loss_val: 1.6045 acc_val: 0.9000 time: 0.0143s split_loss:0.0757 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8724
Epoch: 0118 loss_train: 1.3410 acc_train: 0.9067 loss_val: 1.6024 acc_val: 0.9000 time: 0.0145s split_loss:0.0734 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8834
Epoch: 0119 loss_train: 1.4056 acc_train: 0.9067 loss_val: 1.5997 acc_val: 0.9000 time: 0.0148s split_loss:0.0778 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8120
Epoch: 0120 loss_train: 1.4738 acc_train: 0.9067 loss_val: 1.5974 acc_val: 0.9000 time: 0.0146s split_loss:0.0781 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8241
Epoch: 0121 loss_train: 1.3394 acc_train: 0.9067 loss_val: 1.5946 acc_val: 0.9000 time: 0.0144s split_loss:0.0737 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.8881
Epoch: 0122 loss_train: 1.3268 acc_train: 0.9067 loss_val: 1.5915 acc_val: 0.9000 time: 0.0149s split_loss:0.0727 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.9196
Epoch: 0123 loss_train: 1.4150 acc_train: 0.9067 loss_val: 1.5882 acc_val: 0.9000 time: 0.0180s split_loss:0.0761 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8550
Epoch: 0124 loss_train: 1.3905 acc_train: 0.9067 loss_val: 1.5851 acc_val: 0.9000 time: 0.0184s split_loss:0.0754 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8608
Epoch: 0125 loss_train: 1.3952 acc_train: 0.9067 loss_val: 1.5817 acc_val: 0.9000 time: 0.0175s split_loss:0.0761 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8634
Epoch: 0126 loss_train: 1.2915 acc_train: 0.9067 loss_val: 1.5785 acc_val: 0.9000 time: 0.0185s split_loss:0.0707 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9475
Epoch: 0127 loss_train: 1.3023 acc_train: 0.9067 loss_val: 1.5751 acc_val: 0.9000 time: 0.0186s split_loss:0.0702 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9160
Epoch: 0128 loss_train: 1.3884 acc_train: 0.9067 loss_val: 1.5720 acc_val: 0.9000 time: 0.0187s split_loss:0.0757 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.7999
Epoch: 0129 loss_train: 1.2356 acc_train: 0.9067 loss_val: 1.5690 acc_val: 0.9000 time: 0.0187s split_loss:0.0682 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9475
Epoch: 0130 loss_train: 1.2255 acc_train: 0.9067 loss_val: 1.5660 acc_val: 0.9000 time: 0.0272s split_loss:0.0665 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.9853
Epoch: 0131 loss_train: 1.2836 acc_train: 0.9067 loss_val: 1.5629 acc_val: 0.9000 time: 0.0186s split_loss:0.0704 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9407
Epoch: 0132 loss_train: 1.3697 acc_train: 0.9067 loss_val: 1.5601 acc_val: 0.9000 time: 0.0185s split_loss:0.0744 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.8293
Epoch: 0133 loss_train: 1.2870 acc_train: 0.9067 loss_val: 1.5572 acc_val: 0.9000 time: 0.0184s split_loss:0.0710 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9364
Epoch: 0134 loss_train: 1.2445 acc_train: 0.9067 loss_val: 1.5544 acc_val: 0.9000 time: 0.0185s split_loss:0.0697 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9233
Epoch: 0135 loss_train: 1.3060 acc_train: 0.9067 loss_val: 1.5519 acc_val: 0.9000 time: 0.0184s split_loss:0.0701 split_acc:0.9267 split_recall:0.2143 split_disturb:0.0000 split_auc:0.8167
Epoch: 0136 loss_train: 1.3160 acc_train: 0.9067 loss_val: 1.5501 acc_val: 0.9000 time: 0.0187s split_loss:0.0723 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9107
Epoch: 0137 loss_train: 1.2887 acc_train: 0.9067 loss_val: 1.5485 acc_val: 0.9000 time: 0.0183s split_loss:0.0717 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8655
Epoch: 0138 loss_train: 1.2606 acc_train: 0.9067 loss_val: 1.5467 acc_val: 0.9000 time: 0.0184s split_loss:0.0705 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9212
Epoch: 0139 loss_train: 1.2985 acc_train: 0.9067 loss_val: 1.5447 acc_val: 0.9000 time: 0.0185s split_loss:0.0725 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9396
Epoch: 0140 loss_train: 1.3218 acc_train: 0.9067 loss_val: 1.5429 acc_val: 0.9000 time: 0.0185s split_loss:0.0733 split_acc:0.9067 split_recall:0.0000 split_disturb:0.0000 split_auc:0.8718
Epoch: 0141 loss_train: 1.2082 acc_train: 0.9067 loss_val: 1.5412 acc_val: 0.9000 time: 0.0200s split_loss:0.0679 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9443
Epoch: 0142 loss_train: 1.2180 acc_train: 0.9067 loss_val: 1.5386 acc_val: 0.9000 time: 0.0185s split_loss:0.0687 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9223
Epoch: 0143 loss_train: 1.1407 acc_train: 0.9067 loss_val: 1.5356 acc_val: 0.9000 time: 0.0183s split_loss:0.0647 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.9743
Epoch: 0144 loss_train: 1.2451 acc_train: 0.9067 loss_val: 1.5323 acc_val: 0.9000 time: 0.0183s split_loss:0.0692 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9181
Epoch: 0145 loss_train: 1.2532 acc_train: 0.9067 loss_val: 1.5293 acc_val: 0.9000 time: 0.0154s split_loss:0.0700 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9133
Epoch: 0146 loss_train: 1.2968 acc_train: 0.9133 loss_val: 1.5265 acc_val: 0.9000 time: 0.0151s split_loss:0.0698 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.8451
Epoch: 0147 loss_train: 1.3337 acc_train: 0.9133 loss_val: 1.5236 acc_val: 0.9000 time: 0.0151s split_loss:0.0726 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.8398
Epoch: 0148 loss_train: 1.2560 acc_train: 0.9133 loss_val: 1.5209 acc_val: 0.9000 time: 0.0153s split_loss:0.0687 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.8918
Epoch: 0149 loss_train: 1.2153 acc_train: 0.9067 loss_val: 1.5186 acc_val: 0.9000 time: 0.0150s split_loss:0.0673 split_acc:0.9133 split_recall:0.1429 split_disturb:0.0074 split_auc:0.9569
Epoch: 0150 loss_train: 1.1248 acc_train: 0.9067 loss_val: 1.5158 acc_val: 0.9000 time: 0.0150s split_loss:0.0632 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9223
Epoch: 0151 loss_train: 1.0677 acc_train: 0.9067 loss_val: 1.5122 acc_val: 0.9000 time: 0.0153s split_loss:0.0592 split_acc:0.9467 split_recall:0.4286 split_disturb:0.0000 split_auc:0.9543
Epoch: 0152 loss_train: 1.2317 acc_train: 0.9067 loss_val: 1.5092 acc_val: 0.9000 time: 0.0151s split_loss:0.0691 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.8866
Epoch: 0153 loss_train: 1.2089 acc_train: 0.9133 loss_val: 1.5062 acc_val: 0.9000 time: 0.0150s split_loss:0.0673 split_acc:0.9133 split_recall:0.0714 split_disturb:0.0000 split_auc:0.8766
Epoch: 0154 loss_train: 1.1630 acc_train: 0.9133 loss_val: 1.5039 acc_val: 0.9000 time: 0.0145s split_loss:0.0656 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.9338
Epoch: 0155 loss_train: 1.1700 acc_train: 0.9067 loss_val: 1.5006 acc_val: 0.9000 time: 0.0145s split_loss:0.0661 split_acc:0.9267 split_recall:0.2143 split_disturb:0.0000 split_auc:0.9370
Epoch: 0156 loss_train: 1.1726 acc_train: 0.9133 loss_val: 1.4977 acc_val: 0.9000 time: 0.0144s split_loss:0.0662 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9459
Epoch: 0157 loss_train: 1.2314 acc_train: 0.9067 loss_val: 1.4950 acc_val: 0.9000 time: 0.0143s split_loss:0.0682 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.9254
Epoch: 0158 loss_train: 1.2227 acc_train: 0.9133 loss_val: 1.4922 acc_val: 0.9000 time: 0.0143s split_loss:0.0683 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9133
Epoch: 0159 loss_train: 1.2528 acc_train: 0.9133 loss_val: 1.4896 acc_val: 0.9000 time: 0.0143s split_loss:0.0684 split_acc:0.9067 split_recall:0.1429 split_disturb:0.0147 split_auc:0.8750
Epoch: 0160 loss_train: 1.1546 acc_train: 0.9067 loss_val: 1.4869 acc_val: 0.9000 time: 0.0143s split_loss:0.0645 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.9359
Epoch: 0161 loss_train: 1.0821 acc_train: 0.9133 loss_val: 1.4838 acc_val: 0.9000 time: 0.0144s split_loss:0.0612 split_acc:0.9333 split_recall:0.2857 split_disturb:0.0000 split_auc:0.9533
Epoch: 0162 loss_train: 1.1009 acc_train: 0.9267 loss_val: 1.4808 acc_val: 0.9000 time: 0.0143s split_loss:0.0606 split_acc:0.9400 split_recall:0.3571 split_disturb:0.0000 split_auc:0.9202
Epoch: 0163 loss_train: 1.0264 acc_train: 0.9267 loss_val: 1.4777 acc_val: 0.9000 time: 0.0144s split_loss:0.0555 split_acc:0.9533 split_recall:0.5714 split_disturb:0.0074 split_auc:0.9491
Epoch: 0164 loss_train: 1.1170 acc_train: 0.9133 loss_val: 1.4746 acc_val: 0.9000 time: 0.0144s split_loss:0.0632 split_acc:0.9267 split_recall:0.2143 split_disturb:0.0000 split_auc:0.9779
Epoch: 0165 loss_train: 1.1888 acc_train: 0.9067 loss_val: 1.4720 acc_val: 0.9000 time: 0.0144s split_loss:0.0672 split_acc:0.9133 split_recall:0.1429 split_disturb:0.0074 split_auc:0.9417
Epoch: 0166 loss_train: 1.0848 acc_train: 0.9200 loss_val: 1.4693 acc_val: 0.9000 time: 0.0143s split_loss:0.0583 split_acc:0.9467 split_recall:0.5000 split_disturb:0.0074 split_auc:0.9049
Epoch: 0167 loss_train: 1.0865 acc_train: 0.9133 loss_val: 1.4668 acc_val: 0.9000 time: 0.0142s split_loss:0.0613 split_acc:0.9200 split_recall:0.1429 split_disturb:0.0000 split_auc:0.9480
Epoch: 0168 loss_train: 1.1912 acc_train: 0.9067 loss_val: 1.4647 acc_val: 0.9000 time: 0.0140s split_loss:0.0668 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.8787
Epoch: 0169 loss_train: 1.0433 acc_train: 0.9200 loss_val: 1.4625 acc_val: 0.9000 time: 0.0140s split_loss:0.0586 split_acc:0.9400 split_recall:0.3571 split_disturb:0.0000 split_auc:0.9569
Epoch: 0170 loss_train: 1.1525 acc_train: 0.9067 loss_val: 1.4603 acc_val: 0.9000 time: 0.0139s split_loss:0.0655 split_acc:0.9067 split_recall:0.1429 split_disturb:0.0147 split_auc:0.9475
Epoch: 0171 loss_train: 1.1787 acc_train: 0.9067 loss_val: 1.4584 acc_val: 0.9000 time: 0.0138s split_loss:0.0669 split_acc:0.9133 split_recall:0.1429 split_disturb:0.0074 split_auc:0.9244
Epoch: 0172 loss_train: 1.0537 acc_train: 0.9067 loss_val: 1.4550 acc_val: 0.9000 time: 0.0139s split_loss:0.0594 split_acc:0.9467 split_recall:0.4286 split_disturb:0.0000 split_auc:0.9575
Epoch: 0173 loss_train: 1.1514 acc_train: 0.9067 loss_val: 1.4524 acc_val: 0.9000 time: 0.0139s split_loss:0.0654 split_acc:0.9267 split_recall:0.2143 split_disturb:0.0000 split_auc:0.9207
Epoch: 0174 loss_train: 1.0713 acc_train: 0.9267 loss_val: 1.4493 acc_val: 0.9000 time: 0.0139s split_loss:0.0597 split_acc:0.9400 split_recall:0.3571 split_disturb:0.0000 split_auc:0.9249
Epoch: 0175 loss_train: 1.0212 acc_train: 0.9200 loss_val: 1.4457 acc_val: 0.9000 time: 0.0140s split_loss:0.0577 split_acc:0.9400 split_recall:0.3571 split_disturb:0.0000 split_auc:0.9732
Epoch: 0176 loss_train: 1.0956 acc_train: 0.9200 loss_val: 1.4421 acc_val: 0.9000 time: 0.0140s split_loss:0.0598 split_acc:0.9400 split_recall:0.4286 split_disturb:0.0074 split_auc:0.9160
Epoch: 0177 loss_train: 1.0249 acc_train: 0.9133 loss_val: 1.4394 acc_val: 0.9000 time: 0.0140s split_loss:0.0577 split_acc:0.9333 split_recall:0.4286 split_disturb:0.0147 split_auc:0.9517
Epoch: 0178 loss_train: 1.1997 acc_train: 0.9067 loss_val: 1.4365 acc_val: 0.9000 time: 0.0139s split_loss:0.0665 split_acc:0.9200 split_recall:0.2143 split_disturb:0.0074 split_auc:0.9081
Epoch: 0179 loss_train: 1.1038 acc_train: 0.9133 loss_val: 1.4335 acc_val: 0.9000 time: 0.0140s split_loss:0.0611 split_acc:0.9400 split_recall:0.4286 split_disturb:0.0074 split_auc:0.9259
Epoch: 0180 loss_train: 1.0066 acc_train: 0.9267 loss_val: 1.4307 acc_val: 0.8900 time: 0.0139s split_loss:0.0568 split_acc:0.9400 split_recall:0.3571 split_disturb:0.0000 split_auc:0.9748
Epoch: 0181 loss_train: 0.9539 acc_train: 0.9200 loss_val: 1.4276 acc_val: 0.8900 time: 0.0189s split_loss:0.0521 split_acc:0.9533 split_recall:0.5714 split_disturb:0.0074 split_auc:0.9680
Epoch: 0182 loss_train: 1.0385 acc_train: 0.9267 loss_val: 1.4243 acc_val: 0.8900 time: 0.0183s split_loss:0.0561 split_acc:0.9400 split_recall:0.4286 split_disturb:0.0074 split_auc:0.9459
Epoch: 0183 loss_train: 0.9787 acc_train: 0.9267 loss_val: 1.4210 acc_val: 0.8900 time: 0.0183s split_loss:0.0535 split_acc:0.9533 split_recall:0.5000 split_disturb:0.0000 split_auc:0.9590
Epoch: 0184 loss_train: 0.9950 acc_train: 0.9133 loss_val: 1.4184 acc_val: 0.8900 time: 0.0182s split_loss:0.0543 split_acc:0.9467 split_recall:0.5000 split_disturb:0.0074 split_auc:0.9501
Epoch: 0185 loss_train: 1.0725 acc_train: 0.9133 loss_val: 1.4167 acc_val: 0.8900 time: 0.0184s split_loss:0.0587 split_acc:0.9467 split_recall:0.5714 split_disturb:0.0147 split_auc:0.9249
Epoch: 0186 loss_train: 1.0799 acc_train: 0.9133 loss_val: 1.4160 acc_val: 0.8900 time: 0.0182s split_loss:0.0602 split_acc:0.9200 split_recall:0.2857 split_disturb:0.0147 split_auc:0.9364
Epoch: 0187 loss_train: 1.1126 acc_train: 0.9200 loss_val: 1.4159 acc_val: 0.8900 time: 0.0183s split_loss:0.0620 split_acc:0.9133 split_recall:0.2143 split_disturb:0.0147 split_auc:0.9412
Epoch: 0188 loss_train: 1.0308 acc_train: 0.9267 loss_val: 1.4155 acc_val: 0.8900 time: 0.0182s split_loss:0.0575 split_acc:0.9200 split_recall:0.2857 split_disturb:0.0147 split_auc:0.9485
Epoch: 0189 loss_train: 0.9641 acc_train: 0.9333 loss_val: 1.4145 acc_val: 0.8900 time: 0.0185s split_loss:0.0525 split_acc:0.9533 split_recall:0.5000 split_disturb:0.0000 split_auc:0.9627
Epoch: 0190 loss_train: 1.0371 acc_train: 0.9200 loss_val: 1.4131 acc_val: 0.8900 time: 0.0173s split_loss:0.0577 split_acc:0.9333 split_recall:0.2857 split_disturb:0.0000 split_auc:0.9522
Epoch: 0191 loss_train: 0.9955 acc_train: 0.9200 loss_val: 1.4103 acc_val: 0.8900 time: 0.0144s split_loss:0.0555 split_acc:0.9467 split_recall:0.5000 split_disturb:0.0074 split_auc:0.9648
Epoch: 0192 loss_train: 1.0098 acc_train: 0.9200 loss_val: 1.4067 acc_val: 0.8900 time: 0.0147s split_loss:0.0556 split_acc:0.9400 split_recall:0.4286 split_disturb:0.0074 split_auc:0.9412
Epoch: 0193 loss_train: 1.0624 acc_train: 0.9067 loss_val: 1.4023 acc_val: 0.8900 time: 0.0147s split_loss:0.0610 split_acc:0.9267 split_recall:0.2857 split_disturb:0.0074 split_auc:0.9512
Epoch: 0194 loss_train: 0.9682 acc_train: 0.9200 loss_val: 1.3976 acc_val: 0.8900 time: 0.0146s split_loss:0.0558 split_acc:0.9333 split_recall:0.2857 split_disturb:0.0000 split_auc:0.9800
Epoch: 0195 loss_train: 1.0529 acc_train: 0.9133 loss_val: 1.3913 acc_val: 0.8900 time: 0.0145s split_loss:0.0606 split_acc:0.9267 split_recall:0.2857 split_disturb:0.0074 split_auc:0.9396
Epoch: 0196 loss_train: 0.8835 acc_train: 0.9200 loss_val: 1.3855 acc_val: 0.8900 time: 0.0144s split_loss:0.0517 split_acc:0.9600 split_recall:0.6429 split_disturb:0.0074 split_auc:0.9848
Epoch: 0197 loss_train: 0.9729 acc_train: 0.9267 loss_val: 1.3813 acc_val: 0.8900 time: 0.0147s split_loss:0.0534 split_acc:0.9333 split_recall:0.4286 split_disturb:0.0147 split_auc:0.9475
Epoch: 0198 loss_train: 0.9243 acc_train: 0.9467 loss_val: 1.3782 acc_val: 0.8900 time: 0.0144s split_loss:0.0488 split_acc:0.9467 split_recall:0.5714 split_disturb:0.0147 split_auc:0.9433
Epoch: 0199 loss_train: 0.9361 acc_train: 0.9200 loss_val: 1.3761 acc_val: 0.8900 time: 0.0145s split_loss:0.0501 split_acc:0.9467 split_recall:0.6429 split_disturb:0.0221 split_auc:0.9659
Epoch: 0200 loss_train: 0.9251 acc_train: 0.9200 loss_val: 1.3746 acc_val: 0.8900 time: 0.0152s split_loss:0.0520 split_acc:0.9400 split_recall:0.5714 split_disturb:0.0221 split_auc:0.9575
"""