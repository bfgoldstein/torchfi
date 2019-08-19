import numpy as np
import os
import sys
import numpy as np
import pickle

import torch

def getRecordPrefix(args, dataType, faulty=False):
        if faulty:
                ret =  dataType + '_faulty' +           \
                        '_layer_' + str(args.layer) +        \
                        '_bit_' + str(args.bit) +            \
                        '_epoch_' + str(args.fiEpoch) +      \
                        '_weights_' + str(args.fiWeights) +  \
                        '_features_' + str(args.fiFeats)     
                return ret
        else:
                return dataType + '_golden'
        
def saveRecord(fidPrefixName, record):
        import pickle
        fname = fidPrefixName + "_record.pkl" 
        with open(fname, 'wb') as outFID:
            pickle.dump(record, outFID)
            

def loadRecordsCorrect(dataType, data, nlayers, bit, loc, experiment):
        prefix = experiment + dataType + '_golden'
        recPath = os.path.join(data, prefix)
        print('Loading ' + recPath + "_record.pkl")

        golden = loadRecord(recPath)

        target = []
        for label, acc in golden.targets:
                target.append(label)
        
        predGolden = torch.cat((golden.predictions[0][0], golden.predictions[1][0]))
        for item in golden.predictions[2:]:
                predGolden = torch.cat((predGolden, item[0]))

        correctGolden = predGolden.eq(torch.LongTensor(target))

        prefix = getPrefix(experiment, dataType, nlayers, bit, loc)
        recPath = os.path.join(data, prefix)

        print('Loading ' + recPath + "_record.pkl")

        record = loadRecord(recPath)

        predFaulty = torch.cat((record.predictions[0][0], record.predictions[1][0]))
        for item in record.predictions[2:]:
                predFaulty = torch.cat((predFaulty, item[0]))

        correctFaulty = predFaulty.eq(torch.LongTensor(target))

        return predGolden, predFaulty, target


def loadRecordsLayer(dataType, data, nlayers, bit, loc, experiment):
        prefix = experiment + dataType + '_golden'
        recPath = os.path.join(data, prefix)
        print('Loading ' + recPath + "_record.pkl")

        golden = loadRecord(recPath)

        records = []
        sdcs = []
        acc1s = []

        acc1s.append(golden.acc1)

        for layer in range(0, nlayers):
                prefix = getPrefix(experiment, dataType, layer, bit, loc)
                recPath = os.path.join(data, prefix)

                print('Loading ' + recPath + "_record.pkl")

                record = loadRecord(recPath)

                top1SDC, top5SDC = calculteSDCs(golden.predictions, record.predictions)

                sdcs.append(top1SDC)
                acc1s.append(record.acc1)
                records.append(record)

        return records, sdcs, acc1s


def loadRecordsLayerAvg(dataType, data, nlayers, bit, loc, niter, experiment):
        prefix = experiment + dataType + '_golden'
        recPath = os.path.join(data, prefix)
        print('Loading ' + recPath + "_record.pkl")

        golden = loadRecord(recPath)

        records = []
        sdcs = []
        acc1s = []

        acc1s.append(golden.acc1)

        for layer in range(0, nlayers):
                top1SDCSum = 0
                top5SDCSum = 0
                acc1Sum = 0
                titer = 0
                for iter in range(1, niter + 1):
                        titer += 1
                        prefix = getPrefix(experiment, dataType, layer, bit, loc, iter)
                        recPath = os.path.join(data, prefix)

                        print('Loading ' + recPath + "_record.pkl")
                        try:
                                record = loadRecord(recPath)
                        except FileNotFoundError:
                                print(recPath + "_record.pkl NOT FOUND. Skipping...")
                                titer -= 1
                                continue

                        top1SDC, top5SDC = calculteSDCs(golden.predictions, record.predictions)
                        top1SDCSum += top1SDC
                        top5SDCSum += top5SDC
                        acc1Sum += record.acc1

                top1SDCAvg = float(top1SDCSum) / float(titer)
                top5SDCAvg = float(top5SDCSum) / float(titer)
                acc1Avg = float(acc1Sum) / float(titer)

                sdcs.append(top1SDCAvg)
                acc1s.append(acc1Avg)

        return records, sdcs, acc1s


def loadRecordsBit(dataType, data, layer, nbits, loc, experiment):
        prefix = experiment + dataType + '_golden'
        recPath = os.path.join(data, prefix)
        print('Loading ' + recPath + "_record.pkl")

        golden = loadRecord(recPath)

        records = []
        sdcs = []

        for bit in range(0, nbits):
                prefix = getPrefix(experiment, dataType, layer, bit, loc)
                recPath = os.path.join(data, prefix)

                print('Loading ' + recPath + "_record.pkl")

                record = loadRecord(recPath)

                top1SDC, top5SDC = calculteSDCs(golden.predictions, record.predictions)

                sdcs.append(top1SDC)
                records.append(record)

        return records, sdcs


def loadRecord(fidPrefixName):
    fname = fidPrefixName + "_record.pkl" 
    with open(fname, 'rb') as inFID:
        record = pickle.load(inFID)
        return record


def getPrefix(experiment, dataType, layer, bit, loc, iter=1):
        fidPrefixName = experiment + dataType + '_'
        fidPrefixName += 'layer_' + str(layer) + '_'
        fidPrefixName += 'bit_' + str(bit) + '_' 
        fidPrefixName += 'loc_' + loc + '_' 
        fidPrefixName += 'iter_' + str(iter)
        return fidPrefixName


def calculteSDCs(goldenPred, faultyPred):
    top1Sum = 0
    top5Sum = 0
    for goldenTensor, faultyTensor in zip(goldenPred, faultyPred):
        correct = goldenTensor.ne(faultyTensor)
        top1Sum += correct[:1].view(-1).int().sum(0, keepdim=True)
        for goldenRow, faultyRow in zip(goldenTensor.t(), faultyTensor.t()):
            if goldenRow[0] not in faultyRow:
                top5Sum += 1
    # calculate top1 and top5 SDCs by dividing sum to numBatches * batchSize
    top1SDC = float(top1Sum[0]) / float(len(goldenPred) * len(goldenPred[0][0]))
    top5SDC = float(top5Sum) / float(len(goldenPred) * len(goldenPred[0][0]))
    top1SDC *= 100
    top5SDC *= 100
    return top1SDC, top5SDC


def getAdjacency(correctPred, faultyPred):
    matrixShape = int(max(correctPred)) + 1
    adj = np.zeros((matrixShape, matrixShape))
    errors = 0

    for cpred, fpred in zip(correctPred, faultyPred):
        if cpred > matrixShape or fpred > matrixShape or cpred < 0 or fpred < 0:
            errors += 1
            continue
        else:
            adj[int(cpred)][int(fpred)] += 1

    return adj, errors


def getTopFaulty(adjacency):
    matrixShape = adjacency.shape()
    
    topFaulty = 0
    row = 0
    col = 0

    for i in range(matrixShape[0]):
        for j in range(i + 1, matrixShape[1]):
            if adjacency[i][j] > topFaulty:
                topFaulty = adjacency[i][j]
                row = i
                col = j

    return row, col, topFaulty



class Record(object):

        def __init__(self, model, batch_size, injection=True, fiLayer=None, fiFeatures=None, fiWeights=None, 
                     quantization=False, quant_bfeats=None, quant_bwts=None, quant_baccum=None):
                """Record constructor
                        
                Attributes:
                        model (String): name of the model (arch in torchvision)
                        
                        batch_size (integer): size batch used during each iteration
                        
                        injection (boolean): If True, injection was applied and record represents a faulty run
                        
                        fiFeatures (bool): If True, faults can be applied on input features

                        fiWeights (bool): If True, faults can be applied on weights

                        fiFeatures and fiWeights == False: Faults are applied randomly on input features and weighs
                        
                        fiFeatures (1d ndarray): array of integers with bit flipped position of each batch
                        
                        fiBit (1d ndarray): array of integers with bit flipped position of each batch
                        
                        originalValue (1d ndarray): array of floats with value before injection of each batch
                        
                        fiValue (1d ndarray): array of floats with value after injection of each batch
                        
                        fiLocation (tuple of ints): typle indicating location of fault injection
                        e.g.: (filter, row, column)

                        quantization (boolean): If True, quantization was applied with quant_bfeats, quant_bwts and quant_baccum bits
                         for features, weights and accumulator respectively

                        quant_bfeats (integer): number of bits for input features during quantization
                        
                        quant_bwts (integer): number of bits for weights during quantization
                        
                        quant_baccum (integer): number of bits for accumulators during quantization
                        
                        scores (2d tensor): array of scores (only top 5 values) of each batch

                        predictions (2d tensor): array of labels predicted (only top 5 labels) of each batch
                        
                        targets (2d tensor): array of targets list of each batch. Each target list contains 
                        an integer and float that represents correct label and obtained accuracy respectively
                        
                        targetsGoldenFaulty (2d tensor): array of targets list of each batch. Each target list contains 
                        an integer and float that represents correct label and obtained accuracy respectively. This list 
                        differs from "targets" in respect that the correct predicted label is from the Golden run.
                
                        acc1 (float): float number representing the final top-1 accuracy of the prediction model
                        
                        acc5 (float): float number representing the final top-5 accuracy of the prediction model
                
                        train_losses (1d array): array of losses obtained during each epoch of a model during training phase
                        
                        train_accs (1d array): array of accuracies obtained during each epoch of a model during training phase 
                        
                        test_losses (1d array): array of losses obtained during each epoch of a model during testing phase
                        
                        test_accs (1d array): array of accuracies obtained during each epoch of a model during testing phase
                """

                self.model = model
                self.batch_size = batch_size

                self.injection = injection
                self.fiLayer = fiLayer
                self.fiFeatures = fiFeatures
                self.fiWeights = fiWeights
                self.fiBit = []
                self.originalValue = []
                self.fiValue = []
                self.fiLocation = [] 

                self.quantization = quantization
                self.quant_bfeats = quant_bfeats
                self.quant_bwts = quant_bwts
                self.quant_baccum = quant_baccum

                self.scores = []
                self.predictions = []
                self.targets = []
                self.targetsGoldenFaulty = []
                self.acc1 = 0
                self.acc5 = 0

                # Training data
                self.train_losses = []
                self.train_accs = []
                self.test_losses = []
                self.test_accs = []
                

        def addScores(self, tensors):
                self.scores.append(tensors.cpu())
    
        def addPredictions(self, tensors):
                self.predictions.append(tensors.cpu())

        def addTargets(self, arr):
                for val in arr:
                        self.targets.append(val)
            
        def addTargetsGoldenFaulty(self, arr):
                for val in arr:
                        self.targetsGoldenFaulty.append(val)

        def addFiBit(self, bit):
                # pass
                self.fiBit.append(bit)

        def addOriginalValue(self, value):
                # pass
                self.originalValue.append(value)

        def addFiValue(self, value):
                # pass
                self.fiValue.append(value)

        def addFiLocation(self, loc):
                # pass
                self.fiLocation.append(loc)

        def setAccuracies(self, acc1, acc5):
                self.acc1 = acc1
                self.acc5 = acc5

        def addTrainLosses(self, losses):
                self.train_losses += losses
                
        def addTrainAccs(self, accs):
                self.train_accs += accs
        
        def addTestLosses(self, losses):
                self.test_losses += losses
        
        def addTestAccs(self, accs):
                self.test_accs += accs
                
        def getTargetLabels(self):
                ret = np.zeros(len(self.targets))
                for i, target in enumerate(self.targets):
                        ret[i] = target[0]
                return ret

        def getTop1PredictionLabels(self):
                ret = np.zeros(len(self.predictions))
                for i, pred in enumerate(self.predictions):
                        ret[i] = pred[0]
                return ret

        def getTop1PredictionLabelsBatch(self):
                ret = np.zeros(len(self.predictions) * len(self.predictions[0][0]))
                i = 0
                for batch in self.predictions:
                        for pred in batch[0]:
                                ret[i] = pred
                                i += 1
                return ret