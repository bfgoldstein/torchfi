import sys
import numpy as np
import torch


def calculteSDCs(goldenScores, faultyScores):
    maxk = 5
    
    goldenAcc, goldenPred = goldenScores.topk(maxk, 1, True, True)
    faultyAcc, faultyPred = faultyScores.topk(maxk, 1, True, True)

    top1Sum = 0
    top5Sum = 0
    
    correct = goldenPred.t().ne(faultyPred.t())
    top1Sum += correct[:1].view(-1).int().sum(0, keepdim=True)
    for goldenRow, faultyRow in zip(goldenPred, faultyPred):
        if goldenRow[0] not in faultyRow:
            top5Sum += 1
    top1SDC = float(top1Sum[0]) / float(len(goldenPred))
    top5SDC = float(top5Sum) / float(len(goldenPred))
    top1SDC *= 100
    top5SDC *= 100
    return top1SDC, top5SDC


def main():
    goldenSuffix = sys.argv[1]
    faultySuffix = sys.argv[2]
    nlayers = int(sys.argv[3])

    goldenPrefix = '_score_golden.npz'
    faultyPrefix = '_score_faulty.npz'
    
    goldenScores = stackScores(np.load(goldenSuffix + goldenPrefix))
    goldenScoresTensor = torch.from_numpy(goldenScores)

    # sdcs = {}
    for layer in range(0, nlayers):
        faultyScores = stackScores(np.load(faultySuffix + faultyPrefix))
        faultyScoresTensor = torch.from_numpy(faultyScores)
        # sdcs['layer_' + str(layer)] = 
        top1_sdc, top5_sdc = calculteSDCs(goldenScoresTensor, faultyScoresTensor)
        print('SDCs * SDC@1 {top1_sdc:.3f} SDC@5 {top5_sdc:.3f}'
            .format(top1_sdc=top1_sdc, top5_sdc=top5_sdc))


def stackScores(scores):
    batches = []
    for batchId in scores:
        batches.append(scores[batchId])
    return np.vstack(batches)


if __name__ == "__main__":
    main()