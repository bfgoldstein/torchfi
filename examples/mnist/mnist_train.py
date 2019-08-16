from __future__ import print_function

import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import torchFI as tfi
import util.parser as tfiParser

# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('10.190.0.3', 8097), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class NetDropOut(nn.Module):
    def __init__(self):
        super(NetDropOut, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.dropout(self.conv1(x), training=True))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(F.dropout(self.conv2(x), training=True))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    tfiargs = tfiParser.getParser()
    args = tfiargs.parse_args()
    
    dataset_name = 'MNIST'

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True 
        
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        print("Use GPU: {} for training".format(args.gpu))
    
    normalize_data = ((0.1307,), (0.3081,))
    
    train_loader, test_loader = tfi.loadData(args, dataset_name, normalize_data)
                    
    model = Net()
    fi_model = Net()
    
    if args.gpu is not None:
        model = model.cuda(args.gpu)
        fi_model = fi_model.cuda(args.gpu)
    else:
        model = model.cpu()
        fi_model = fi_model.cpu()
        
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    fi_optimizer = optim.SGD(fi_model.parameters(), lr=args.lr, momentum=args.momentum)
    
    tfi.fi_train(args, dataset_name, model, optimizer, fi_model, fi_optimizer, criterion, train_loader, test_loader)

        
if __name__ == '__main__':
    main()
