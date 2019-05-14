import os
import sys
import numpy as np
import pickle

import torch


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

        # return correctGolden, correctFaulty, target
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
                # print("Golden: ", golden.acc1, golden.acc5)
                # print("Faulty: ", record.acc1, record.acc5)
                # print("SDCs: ", top1SDC, top5SDC)

                # sdcs.append((top1SDC, top5SDC))
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
                # print("Golden: ", golden.acc1, golden.acc5)
                # print("Faulty: ", record.acc1, record.acc5)
                # print("SDCs: ", top1SDC, top5SDC)

                # sdcs.append((top1SDC, top5SDC))
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