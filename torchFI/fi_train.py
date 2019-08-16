from __future__ import print_function

import os
import random
import time
import warnings
import sys
import math

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import torchFI as tfi
from torchFI.injection import FI
import util.parser as tfiParser
from util.log import logConfig
from util.record import *
from util.plot.train_report import *

from torch.utils.tensorboard import SummaryWriter


def loadData(args, dataset_name, normalize_data):
    from torchvision import datasets, transforms
    
    data_class = {'CIFAR10':datasets.CIFAR10,
                  'MNIST':datasets.MNIST}
    
    normalize = transforms.Normalize(normalize_data[0], normalize_data[1])
    
    train_loader = torch.utils.data.DataLoader(
        data_class[dataset_name](args.data, train=True, download=True,
                       transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=(args.gpu is not None))
    
    test_loader = torch.utils.data.DataLoader(
        data_class[dataset_name](args.data, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                    ])),
                    batch_size=args.test_batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(args.gpu is not None))
    
    return train_loader, test_loader


def adjust_learning_rate(optimizer, epoch, gamma):
    if epoch in [150, 225]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            
            
def train(args, model, train_loader, criterion, optimizer, epoch, fi=None, batchFault=None):
    model.train()

    epoch_loss = []
    epoch_acc = []
    correct = 0
    total_count = 0

    if fi is not None:
        fi.injectionMode = False
        
    for batch_idx, (input, target) in enumerate(train_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
        if fi is not None and batchFault is not None:
            if epoch == fi.epoch and batch_idx in batchFault:
                fi.injectionMode = True
                
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_count += len(input)
        accuracy = correct / total_count
      
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    epoch_loss.append(loss.item())
    epoch_acc.append(accuracy)
        
    return epoch_loss, epoch_acc


def test(args, model, criterion, test_loader, fi=None):
    model.eval()
    # test_loss = []
    test_loss_avg = 0
    correct = 0
    accs = []
    batch_count = 0
    
    if fi is not None:
        fi.injectionMode = False
    
    with torch.no_grad():
        for input, target in test_loader:
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                
            output = model(input)
            # test_loss.append(criterion(output, target).item())
            test_loss_avg += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            batch_count += len(input)
            accs.append(correct / batch_count)

    test_loss_avg /= len(test_loader.dataset)
    acc_avg = correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_avg, correct, len(test_loader.dataset), 100. * acc_avg))
    
    return test_loss_avg, acc_avg


def fi_train(args, dataset_name, model, optimizer, fi_model, fi_optimizer, criterion, train_loader, test_loader):
    
    if args.log_prefix is not None and args.log_path is not None:
        writer = SummaryWriter(log_dir=args.log_path + '/tensorboard_runs/' + args.log_prefix)
    else:
        writer = SummaryWriter()
    
   
    
    ##### Fault Injection #########
    batchFault = np.random.choice(math.floor(len(train_loader) / args.batch_size), 1, replace=False)
 
    record = Record(dataset_name, batch_size=args.batch_size, injection=args.injection, fiLayer=args.layer, fiFeatures=args.fiFeats, fiWeights=args.fiWeights)
       
    fi = FI(fi_model, fiMode=args.injection, fiLayer=args.layer, fiBit=args.bit, fiepoch=args.fiEpoch, 
            fiFeatures=args.fiFeats, fiWeights=args.fiWeights, quantType='SYMMETRIC')
    
    fi.traverseModel(fi_model)
    ##############################

    print(model._modules.items())
    print(fi_model._modules.items())
    print('batchFault= ' + str(batchFault))
    print('')
    
    # Golden Run
    if args.golden:
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        print('Golden Training \n')
        
        for epoch in range(1, args.epochs + 1):
            if dataset_name == 'CIFAR10':
                adjust_learning_rate(optimizer, epoch, args.gamma)
            epoch_loss, epoch_acc = tfi.train(args, model, train_loader, criterion, optimizer, epoch)
            train_losses += epoch_loss
            train_accs += epoch_acc
            
            test_loss, test_acc = tfi.test(args, model, criterion, test_loader)
            test_losses += [test_loss]
            test_accs += [test_acc]
                   
        if args.record_prefix is not None:
            saveRecord(args.record_prefix + getRecordPrefix(args, 'fp32', faulty=False), record) 
        
    # Faulty Run
    if args.faulty:    
        fi_train_losses = []
        fi_train_accs = []
        fi_test_losses = []
        fi_test_accs = []
        
        print('Faulty Training \n')
        
        for epoch in range(1, args.epochs + 1):
            if dataset_name == 'CIFAR10':
                adjust_learning_rate(fi_optimizer, epoch, args.gamma)
            epoch_loss, epoch_acc = tfi.train(args, fi_model, train_loader, criterion, fi_optimizer, epoch,
                                          fi, batchFault)
            fi_train_losses += epoch_loss
            fi_train_accs += epoch_acc
            record.addTrainLosses(epoch_loss)
            record.addTrainAccs(epoch_acc)
            
            test_loss, test_acc = tfi.test(args, fi_model, criterion, test_loader, fi)
            fi_test_losses += [test_loss]
            fi_test_accs += [test_acc]
            
            record.addTestLosses([test_loss])
            record.addTestAccs([test_acc])
            
            for tag, value in fi_model.named_parameters():
                tag = tag.replace('.', '/')                                         
                writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        
        if args.record_prefix is not None:
            saveRecord(args.record_prefix + getRecordPrefix(args, 'fp32', faulty=True), record)

    for i in range(0, args.epochs):
        writer.add_scalars('Loss/train', { 'golden' : train_losses[i], 
                                          'faulty' : fi_train_losses[i]}, i + 1)
        writer.add_scalars('Loss/test', { 'golden' : test_losses[i], 
                                         'faulty' : fi_test_losses[i]}, i + 1)
        writer.add_scalars('Accuracy/train', { 'golden' : train_accs[i], 
                                              'faulty' : fi_train_accs[i]}, i + 1)
        writer.add_scalars('Accuracy/test', { 'golden' : test_accs[i], 
                                             'faulty' : fi_test_accs[i]}, i + 1)
        
    if args.plot is not None:
        plot_losses(args.plot, train_losses, fi_train_losses, test_losses, fi_test_losses)

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model + ".pt")
        
    writer.close()