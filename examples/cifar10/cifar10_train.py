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

# classes = ('plane', 'car', 'bird', 'cat',
#         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    tfiargs = tfiParser.getParser()
    args = tfiargs.parse_args()
    
    dataset_name = 'CIFAR10'
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True 
        
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        print("Use GPU: {} for training".format(args.gpu))
    
    normalize_data = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    train_loader, test_loader = tfi.loadData(args, dataset_name, normalize_data)
                    
    model = Net()
    fi_model = Net()
    
    if args.gpu is not None:
        model = model.cuda(args.gpu)
        fi_model = fi_model.cuda(args.gpu)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fi_optimizer = optim.SGD(fi_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    tfi.fi_train(args, dataset_name, model, optimizer, fi_model, fi_optimizer, criterion, train_loader, test_loader)

  
if __name__ == '__main__':
    main()

