import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl

import torch


def plotBarDeltaTop12(delta_correct, delta_miss, deltaFigName, deltaPlotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    y = np.arange(0, len(delta_correct))

    plt.barh(y, delta_correct, align='center', color='b', label='Correct Prediction')
    
    y = np.arange(y[-1], y[-1] + len(delta_miss))
    plt.barh(y, delta_miss, align='center', color='r', label='Miss Prediction')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Scaled ' + r'$\Delta$' + ' (golden - faulty)')
    ax.set_ylabel('# Input Sample')
    ax.set_title(r'$\Delta$' + ' score' + '\n ' + deltaPlotTitle)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


    fig.tight_layout()
    plt.savefig(deltaFigName + '_deltas.png', dpi = 200)
    # plt.show()


def plotBarSDCs(sdcs, numLayers, plotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    x_names = []
    x_pos_names = []

    y_pos = 0
    
    top1, top5 = sdcs['layer_' + str(0)]
    x_names.append('layer ' + str(0))
    x_pos_names.append(float(y_pos) + 0.5)
    plt.bar(y_pos, top1, align='center', color='b', label='SDC@1')
    plt.bar(y_pos + 1, top5, align='center', color='r', label='SDC@5')
    
    for layer in range(1, numLayers):
        y_pos += 3
        top1, top5 = sdcs['layer_' + str(layer)]
        x_names.append('layer ' + str(layer))
        x_pos_names.append(float(y_pos) + 0.5)
        plt.bar(y_pos, top1, align='center', color='b')
        plt.bar(y_pos + 1, top5, align='center', color='r')
    
    ax.set_ylabel('SDCs Prob.')
    ax.set_title('SDCs' + ' per Layer' + '\n ' + plotTitle)

    plt.xticks(x_pos_names, tuple(x_names), rotation=45)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


    fig.tight_layout()
    plt.savefig(plotTitle + '_sdcs.png', dpi = 200)
    # plt.show()


def plotLineMinPredict(minPredict, numLayers, plotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    x_names = []
    x_pos = np.arange(numLayers)

    avgPredScoreGoldenList = []
    avgPredScoreFaultyList = []
    avgMissScoreGoldenList = []
    avgMissScoreFaultyList = []
    for layer in range(0, numLayers):
        x_names.append('layer ' + str(layer))
        minpred = minPredict['layer_' + str(layer)]
        avgPredScoreGoldenList.append(minpred[0])
        avgPredScoreFaultyList.append(minpred[1])
        avgMissScoreGoldenList.append(minpred[2])
        avgMissScoreFaultyList.append(minpred[3])
  
    plt.plot(x_pos, avgPredScoreGoldenList, 'bo-', label='Pred. Golden')
    plt.plot(x_pos, avgPredScoreFaultyList, 'g^--', label='Pred. Faulty')
    plt.plot(x_pos, avgMissScoreGoldenList, 'rs-.', label='Miss Golden')
    plt.plot(x_pos, avgMissScoreFaultyList, 'm*:', label='Miss Faulty')
    
    ax.set_ylim(0, 10)
    ax.set_ylabel('Avg. Top@1 - Top@2')
    ax.set_title('Minimum Delta' + ' per Layer' + '\n ' + plotTitle)

    plt.xticks(x_pos, tuple(x_names), rotation=45)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


    fig.tight_layout()
    plt.savefig(plotTitle + '_minPred.png', dpi = 200)
    # plt.show()


def featScale(correct, miss):
    arr = np.append(correct, miss)
    max = np.amax(arr)
    min = np.amin(arr)

    return (correct - min) / float(max - min), (miss - min) / float(max - min)


def minMissPredict(goldenAccT, goldenPredT, faultyAccT, faultyPredT):
    
    goldenDelta = goldenAccT[:1].sub(goldenAccT[1:2])
    faultyDelta = faultyAccT[:1].sub(faultyAccT[1:2])

    missTop1 = goldenPredT.ne(faultyPredT)[:1]
    numMiss = missTop1.view(-1).int().sum(0, keepdim=True)
    correctTop1 = goldenPredT.eq(faultyPredT)[:1]
    numCorrect = correctTop1.view(-1).int().sum(0, keepdim=True)

    avgMissScoreGolden = goldenDelta.mul(missTop1.float()).view(-1).float().sum(0, keepdim=True) / float(numMiss)
    avgMissScoreFaulty = faultyDelta.mul(missTop1.float()).view(-1).float().sum(0, keepdim=True) / float(numMiss)
    
    avgPredScoreGolden = goldenDelta.mul(correctTop1.float()).view(-1).float().sum(0, keepdim=True) / float(numCorrect)
    avgPredScoreFaulty = faultyDelta.mul(correctTop1.float()).view(-1).float().sum(0, keepdim=True) / float(numCorrect)

    print('Min Delta Pred * Golden {avgGolden:.3f} Faulty {avgFaulty:.3f}'
        .format(avgGolden=avgPredScoreGolden[0], avgFaulty=avgPredScoreFaulty[0]))

    print('Min Delta MissPred * Golden {avgGolden:.3f} Faulty {avgFaulty:.3f}'
        .format(avgGolden=avgMissScoreGolden[0], avgFaulty=avgMissScoreFaulty[0]))

    return avgPredScoreGolden, avgPredScoreFaulty, avgMissScoreGolden, avgMissScoreFaulty



def calculateDeltas(goldenAccT, goldenPredT, faultyAccT, faultyPredT, faultyScores):
    delta_miss = []
    delta_correct = []

    for idx in range(0, len(goldenPredT[0])):
        if goldenPredT[0][idx] == faultyPredT[0][idx]:
            delta = goldenAccT[0][idx] - faultyAccT[0][idx]
            delta_correct.append(delta)
        else:
            delta = goldenAccT[0][idx] - faultyScores[idx][goldenPredT[0][idx]]
            delta_miss.append(delta)
    
    return np.asarray(delta_correct, dtype=np.float32), np.asarray(delta_miss, dtype=np.float32)


def calculteSDCs(goldenPred, faultyPred):
    top1Sum = 0
    top5Sum = 0
    
    miss = goldenPred.t().ne(faultyPred.t())
    top1Sum += miss[:1].view(-1).int().sum(0, keepdim=True)
    for goldenRow, faultyRow in zip(goldenPred, faultyPred):
        if goldenRow[0] not in faultyRow:
            top5Sum += 1
    top1SDC = float(top1Sum[0]) / float(len(goldenPred))
    top5SDC = float(top5Sum) / float(len(goldenPred))
    top1SDC *= 100
    top5SDC *= 100
    return top1SDC, top5SDC


def stackScores(scores):
    values = list(scores.values())
    return np.asarray(values, dtype=np.float32)


def main():
    goldenSuffix = sys.argv[1]
    faultySuffix = sys.argv[2]
    layer = int(sys.argv[3])
    
    maxk = 5
    
    goldenPrefix = '_score_golden.npz'
    faultyPrefix = '_bit_rand_loc_rand_iter_1_score_faulty.npz'
    
    goldenScores = stackScores(np.load(goldenSuffix + goldenPrefix))
    goldenScoresTensor = torch.from_numpy(goldenScores)

    goldenAcc, goldenPred = goldenScoresTensor.topk(maxk, 1, True, True)

    sdcs = {}

    faultyScores = stackScores(np.load(faultySuffix + str(layer) + faultyPrefix))
    faultyScoresTensor = torch.from_numpy(faultyScores)
    
    faultyAcc, faultyPred = faultyScoresTensor.topk(maxk, 1, True, True)

    minMissPredict(goldenAcc.t(), goldenPred.t(), faultyAcc.t(), faultyPred.t())

    sdcs['layer_' + str(layer)] = calculteSDCs(goldenPred, faultyPred)
    print('SDCs * SDC@1 {sdc[0]:.3f} SDC@5 {sdc[1]:.3f}'
        .format(sdc=sdcs['layer_' + str(layer)]))

    # delta_correct, delta_miss = calculateDeltas(goldenAcc.t(), goldenPred.t(), faultyAcc.t(), faultyPred.t(), faultyScoresTensor)

    # delta_correct, delta_miss = featScale(delta_correct, delta_miss)

    # plotBarDeltaTop12(delta_correct, delta_miss, goldenSuffix, goldenSuffix)
    
    # print(delta_correct)
    # print(delta_miss)

def mainLayers():
    goldenSuffix = sys.argv[1]
    faultySuffix = sys.argv[2]
    project = sys.argv[3]
    numLayers = int(sys.argv[4])
    
    maxk = 5
    
    goldenPrefix = '_score_golden.npz'
    faultyPrefix = '_bit_rand_loc_rand_iter_1_score_faulty.npz'
    
    goldenScores = stackScores(np.load(goldenSuffix + goldenPrefix))
    goldenScoresTensor = torch.from_numpy(goldenScores)

    goldenAcc, goldenPred = goldenScoresTensor.topk(maxk, 1, True, True)

    sdcs = {}
    minPredict = {}
    for layer in range(0, numLayers):
        faultyScores = stackScores(np.load(faultySuffix + str(layer) + faultyPrefix))
        faultyScoresTensor = torch.from_numpy(faultyScores)
        
        faultyAcc, faultyPred = faultyScoresTensor.topk(maxk, 1, True, True)

        sdcs['layer_' + str(layer)] = calculteSDCs(goldenPred, faultyPred)
        print('SDCs * SDC@1 {sdc[0]:.3f} SDC@5 {sdc[1]:.3f}'
            .format(sdc=sdcs['layer_' + str(layer)]))

        # minPredict['layer_' + str(layer)] = minMissPredict(goldenAcc.t(), goldenPred.t(), faultyAcc.t(), faultyPred.t())
        

        # delta_correct, delta_miss = calculateDeltas(goldenAcc.t(), goldenPred.t(), faultyAcc.t(), faultyPred.t(), faultyScoresTensor)

        # delta_correct, delta_miss = featScale(delta_correct, delta_miss)

        # plotBarDeltaTop12(delta_correct, delta_miss, goldenSuffix, goldenSuffix)
        
        # print(delta_correct)
        # print(delta_miss)
    plotBarSDCs(sdcs, numLayers, project)
    # plotLineMinPredict(minPredict, numLayers, project)


if __name__ == "__main__":
    mainLayers()
    # main()