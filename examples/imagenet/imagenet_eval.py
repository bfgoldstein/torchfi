import argparse
import os
import random
import shutil
import time
import warnings
import sys
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import torchFI as tfi
from torchFI.injection import FI
import util.parser as tfiParser
from util.log import logConfig
from util.record import *
from util.tensor import *

# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('10.190.0.3', 8097), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


def createModel(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if args.pruned:
            checkpoint = torch.load(args.pruned_file, map_location='cpu')#'gpu' if args.gpu is not None else 'cpu')
            state_dict = checkpoint['state_dict']
            if args.arch.startswith('alexnet'):
                model.features = torch.nn.DataParallel(model.features)
                model.load_state_dict(state_dict)
            else:
                loadStateDictModel(model, state_dict)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    return model


def loadData(args):
    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(args.gpu is not None))
    return val_loader
    
def main():
    tfiargs = tfiParser.getParser()
    args = tfiargs.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    if args.gpu is not None:
        ngpus_per_node = torch.cuda.device_count()
        main_gpu_worker(args.gpu, ngpus_per_node, args)
    else:
        main_cpu_worker(args)

def main_gpu_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for inference".format(args.gpu))

    model = createModel(args)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    val_loader = loadData(args)
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


def main_cpu_worker(args):
    model = createModel(args)

    # DataParallel will divide and allocate batch_size to all available CPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cpu()
    else:
        # Model are saved into self.module when using DataParellel with CPU only
        model = torch.nn.DataParallel(model).module

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    val_loader = loadData(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


def validate(val_loader, model, criterion, args):
    displayConfig(args)
    
    # switch to evaluate mode
    model.eval()

    if args.pruned:
        _, _, sparsity = countZeroWeights(model)
        logConfig("sparsity", "{}".format(sparsity))
        numBatchFault = math.ceil(len(val_loader) * (1 - sparsity))
        batchFault  = np.random.choice(len(val_loader), numBatchFault, replace=False)

    traverse_time = AverageMeter()
    end = time.time()
    
    record = Record(model=args.arch, batch_size=args.batch_size, 
                    injection=args.injection, fiLayer=args.layer, 
                    fiFeatures=args.fiFeats, fiWeights=args.fiWeights)

    # applying faulty injection scheme
    fi = FI(model, record, fiMode=args.injection, fiLayer=args.layer, fiBit=args.bit, 
            fiFeatures=args.fiFeats, fiWeights=args.fiWeights, log=args.log)

    fi.traverseModel(model)

    if args.quantize:
        quantizer = tfi.FIPostTrainLinearQuantizer(model,
                                                mode=args.quant_mode,
                                                bits_activations=args.quant_bacts,
                                                bits_parameters=args.quant_bwts,
                                                bits_accum=args.quant_baccum,
                                                per_channel_wts=args.quant_channel,
                                                clip_acts=args.quant_cacts,
                                                model_activation_stats=args.quant_stats_file,
                                                clip_n_stds=args.quant_cnstds,
                                                scale_approx_mult_bits=args.quant_scalebits)

        quantizer.prepare_model()   
        # model = quantizer.model
        if args.faulty:
            fi.setQuantParams(args)
        record.setQuantParams(args)
    
    model.eval()
    if args.gpu is not None:
        model = model.cuda()
        
    traverse_time.update(time.time() - end)

    print(model._modules.items())

    batch_time = AverageMeter()
    sdcs = SDCMeter()

    # Golden Run
    if args.golden:

        golden_time = AverageMeter()
        golden_end = time.time()
        
        top1_golden = AverageMeter()
        top5_golden = AverageMeter()

        fi.injectionMode = False
        
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                
                # compute output
                output = model(input)
                
                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_golden.update(acc1[0], input.size(0))
                top5_golden.update(acc5[0], input.size(0))

                sdcs.updateGoldenData(output)

                scores, predictions = topN(output, target, topk=(1,5))
                
                sdcs.updateGoldenBatchPred(predictions)
                sdcs.updateGoldenBatchScore(scores)
                
                record.addPredictions(predictions)
                record.addTargets(correctPred(output, target))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Golden Run: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Acc@1 {top1_golden.val:.3f} ({top1_golden.avg:.3f})\t'
                        'Acc@5 {top5_golden.val:.3f} ({top5_golden.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, top1_golden=top1_golden, top5_golden=top5_golden))
                # break
        golden_time.update(time.time() - golden_end)

    batch_time.reset()

    # Faulty Run
    if args.faulty:

        faulty_time = AverageMeter()
        faulty_end = time.time()

        top1_faulty = AverageMeter()
        top5_faulty = AverageMeter()
        

        with torch.no_grad():
            end = time.time()
           
            for i, (input, target) in enumerate(val_loader):
                fi.injectionMode = True
                
                if args.pruned and args.prune_compensate:
                    if i in batchFault:
                        fi.injectionMode = True
                    else:
                        fi.injectionMode = False
                        
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(input)
                
                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_faulty.update(acc1[0], input.size(0))
                top5_faulty.update(acc5[0], input.size(0))

                sdcs.updateFaultyData(output)

                scores, predictions = topN(output, target, topk=(1,5))
                
                sdcs.updateFaultyBatchPred(predictions)
                sdcs.updateFaultyBatchScore(scores)

                record.addPredictions(predictions)
                record.addTargets(correctPred(output, target))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Faulty Run: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Acc@1 {top1_faulty.val:.3f} ({top1_faulty.avg:.3f})\t'
                        'Acc@5 {top5_faulty.val:.3f} ({top5_faulty.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, top1_faulty=top1_faulty, top5_faulty=top5_faulty))
                # break
        faulty_time.update(time.time() - faulty_end)

    if args.golden:
        print('Golden Run * Acc@1 {top1_golden.avg:.3f} Acc@5 {top5_golden.avg:.3f}'
            .format(top1_golden=top1_golden, top5_golden=top5_golden))
        record.setAccuracies(float(top1_golden.avg), float(top5_golden.avg))

    if args.faulty:
        print('Faulty Run * Acc@1 {top1_faulty.avg:.3f} Acc@5 {top5_faulty.avg:.3f}'
            .format(top1_faulty=top1_faulty, top5_faulty=top5_faulty))
        record.setAccuracies(float(top1_faulty.avg), float(top5_faulty.avg))

    if args.golden and args.faulty:
        print('Acc@1 {top1_diff:.3f} Acc@5 {top5_diff:.3f}'
            .format(top1_diff=(top1_golden.avg - top1_faulty.avg), top5_diff=(top5_golden.avg - top5_faulty.avg)))        
        sdcs.calculteSDCs()
        print('SDCs * SDC@1 {sdc.top1SDC:.3f} SDC@5 {sdc.top5SDC:.3f}'
            .format(sdc=sdcs))

    if args.record_prefix is not None:
        prate = None
        if args.pruned:
            prate = getSparsity(model, norm=True)
        saveRecord(args.record_prefix + getRecordPrefix(args, faulty=args.faulty, quantized=args.quantize, pruned=args.pruned, pruned_rate=prate), record)

    print('Traverse Time {traverse_time.val:.3f}\t'.format(
                traverse_time=traverse_time))
    if args.golden:
        print('Golden Time {golden_time.val:.3f}\t'.format(
                golden_time=golden_time))
    if args.faulty:
        print('Faulty Time {faulty_time.val:.3f}\t'.format(
                faulty_time=faulty_time))
        print('Number of faults applied: {}'.format(int(fi.numInjections)))
    
    return

# TODO: Add GNMT arguments
def displayConfig(args):
    # loging configs to screen
    from util.log import logConfig
    logConfig("model", "{}".format(args.arch))
    logConfig("quantization", "{}".format(args.quantize))
    if args.quantize:
        logConfig("mode", "{}".format(args.quant_mode))
        logConfig("# bits features", "{}".format(args.quant_bacts))
        logConfig("# bits weights", "{}".format(args.quant_bwts))
        logConfig("# bits accumulator", "{}".format(args.quant_baccum))
        logConfig("clip-acts", "{}".format(args.quant_cacts))
        logConfig("per-channel-weights", "{}".format(args.quant_channel))
        logConfig("model-activation-stats", "{}".format(args.quant_stats_file))
        logConfig("clip-n-stds", "{}".format(args.quant_cnstds))
        logConfig("scale-approx-mult-bits", "{}".format(args.quant_scalebits))
    logConfig("injection", "{}".format(args.injection))
    if args.injection:
        logConfig("layer", "{}".format(args.layer))
        logConfig("bit", "{}".format(args.bit))
        logConfig("location:", "  ")
        logConfig("\t features ", "{}".format(args.fiFeats))
        logConfig("\t weights ", "{}".format(args.fiWeights))
        if not(args.fiFeats ^ args.fiWeights): 
            logConfig(" ", "Setting random mode.")
    logConfig("pruned", "{}".format(args.pruned))
    logConfig("prune compensate", "{}".format(args.prune_compensate))
    if args.pruned:
        logConfig("checkpoint from ", "{}".format(args.pruned_file))
    logConfig("batch size", "{}".format(args.batch_size))
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)


class SDCMeter(object):
    """Stores the SDCs probabilities"""
    def __init__(self):
        self.reset()

    def updateAcc(self, acc1, acc5):
        self.acc1 = acc1
        self.acc5 = acc5

    def updateGoldenData(self, scoreTensors):
        for scores in scoreTensors.cpu().numpy():
            self.goldenScoresAll.append(scores)

    def updateFaultyData(self, scoreTensors):
        for scores in scoreTensors.cpu().numpy():
            self.faultyScoresAll.append(scores)
    
    def updateGoldenBatchPred(self, predTensors):
        self.goldenPred.append(predTensors)

    def updateFaultyBatchPred(self, predTensors):
        self.faultyPred.append(predTensors)

    def updateGoldenBatchScore(self, scoreTensors):
        self.goldenScores.append(scoreTensors)

    def updateFaultyBatchScore(self, scoreTensors):
        self.faultyScores.append(scoreTensors)

    def calculteSDCs(self):
        top1Sum = 0
        top5Sum = 0
        for goldenTensor, faultyTensor in zip(self.goldenPred, self.faultyPred):
            correct = goldenTensor.ne(faultyTensor)
            top1Sum += correct[:1].view(-1).int().sum(0, keepdim=True)
            for goldenRow, faultyRow in zip(goldenTensor.t(), faultyTensor.t()):
                if goldenRow[0] not in faultyRow:
                    top5Sum += 1
        # calculate top1 and top5 SDCs by dividing sum to numBatches * batchSize
        self.top1SDC = float(top1Sum[0]) / float(len(self.goldenPred) * len(self.goldenPred[0][0]))
        self.top5SDC = float(top5Sum) / float(len(self.goldenPred) * len(self.goldenPred[0][0]))
        self.top1SDC *= 100
        self.top5SDC *= 100

    def reset(self):
        self.acc1 = 0
        self.acc5 = 0
        self.top1SDC = 0.0
        self.top5SDC = 0.0
        self.goldenPred = []
        self.faultyPred = []
        self.goldenScores = []
        self.faultyScores = []
        self.goldenScoresAll = []
        self.faultyScoresAll = []


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # t() == transpose tensor
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def correctPred(output, target):
    """Return the accuracy of the expected label"""
    with torch.no_grad():
        res = []
        for out, label in zip(output, target):
            acc = out[label]
            res.append([int(label.cpu()), float(acc.cpu())])
    return res


def topN(output, target, topk=(1,)):
    """Return label prediction from top 5 classes"""
    with torch.no_grad():
        maxk = max(topk)
        scores, pred = output.topk(maxk, 1, True, True)
    return scores.t(), pred.t()

        
def loadStateDictModel(model, state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
  

def getSparsity(model, norm=False):
    _, _, sparsity = countZeroWeights(model)
    if norm:
        sparsity = np.floor(sparsity *  100)
    return sparsity
        
          
if __name__ == '__main__':
    main()