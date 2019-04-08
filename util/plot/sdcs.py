import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl

import torch

from util import *

#resnet50_fp32_layer_9_bit_6_loc_weights_iter_1_record.pkl

parser = argparse.ArgumentParser(description='TorchFI Plot')
parser.add_argument('--original', metavar='DIR', dest='origData',
                    help='path to records')
parser.add_argument('--pruned', metavar='DIR', dest='prunedData', 
                    help='path to pruned records')
parser.add_argument('-feats', '--features', dest='features', action='store_true',
                    help='plot data from FI on features/activations')
parser.add_argument('-wts', '--weights', dest='weights', action='store_true',
                    help='plot data from FI on weights')
parser.add_argument('--layers', default=54, dest='nlayers', type=int,
                    help='number of layers to plot')
parser.add_argument('--bit', default=54, dest='bit', type=int,
                    help='bit position to plot')
parser.add_argument('--data_type', default='fp32', dest='dataType', type=str,
                    help='data type to plot')                    
parser.add_argument('--plotLayer', dest='perlayer', action='store_true',
                    help='plot data per Layer')
parser.add_argument('--plotBit', dest='perbit', action='store_true',
                    help='plot data per Bit')
parser.add_argument('--acc', dest='acc', action='store_true',
                    help='plot sdcs vs acc@1')
parser.add_argument('--scores', dest='scores', action='store_true',
                    help='plot scores from golden vs faulty')

def main():
    args = parser.parse_args()
    
    if args.weights:
        loc = 'weights'
    else:
        loc = 'features'

    if args.perlayer:
        sdcsTypes = []
        # dataTypes = ['fp32', 'int16', 'int8']
        dataTypes = ['int16', 'int8']
        for dataType in dataTypes:
            origDataPath = os.path.join(args.origData, dataType)
            prunedDataPath = os.path.join(args.prunedData, dataType)
            _, sdcsOriginal, acc1sOriginal = loadRecordsLayer(dataType, origDataPath, args.nlayers, args.bit, loc, 'resnet50_')
            _, sdcsPruned, accs1sPruned = loadRecordsLayer(dataType, prunedDataPath, args.nlayers, args.bit, loc, 'resnet50_pruned_')
            perLayer(sdcsOriginal, sdcsPruned, 'resnet50_', dataType, loc, args.bit)
            perLayerBar(sdcsOriginal, sdcsPruned, 'resnet50_', dataType, loc, args.bit)
            if args.acc:
                perLayerVsAcc(sdcsOriginal, sdcsPruned, acc1sOriginal, accs1sPruned, 'resnet50_', dataType, loc, args.bit)

    if args.perbit:
        sdcsTypes = []
        dataTypes = [('fp32', 32), ('int16', 16), ('int8', 8)]
        for dataType, nbits in dataTypes:
            dataPath = os.path.join(args.origData, dataType)
            _, sdcs = loadRecordsBit(dataType, dataPath, args.nlayers, nbits, loc, 'resnet50_')
            sdcsTypes.append(sdcs)
        perBit(sdcsTypes, 'resnet50_', args.nlayers, loc)

    if args.scores:
        dataTypes = ['fp32', 'int16', 'int8']
        for dataType in dataTypes:
            dataPath = os.path.join(args.origData, dataType)
            correctGolden, correctFaulty, target = loadRecordsCorrect(dataType, dataPath, args.nlayers, args.bit, loc, 'resnet50_')
            goldenFaultyScores(correctGolden, correctFaulty, target, 'resnet50_', dataType, loc, args.nlayers, args.bit)


def perLayer(sdcsOriginal, sdcsPruned, fname, dataType, loc, bit):
    cmap = plt.get_cmap('tab10')

    x_pos = np.arange(len(sdcsOriginal))

    fig, axarr = plt.subplots(2, figsize=(10,8), sharex=True, sharey=True)
    
    color = cmap(0)
    axarr[0].plot(x_pos, sdcsOriginal, 'o-', c=color, label=dataType + ' original')
    axarr[0].grid(True, linestyle='dotted')
    axarr[0].set_ylabel('SDC Probability')
    # axarr[0].legend(loc='upper right', frameon=False)

    color = cmap(1)
    axarr[1].plot(x_pos, sdcsPruned, 'p-', c=color, label=dataType + ' pruned')
    axarr[1].grid(True, linestyle='dotted')
    axarr[1].set_ylabel('SDC Probability')
    # axarr[1].legend(loc='upper right', frameon=False)
    axarr[1].set_xlabel('Layers')
    
    axarr[0].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=1)
    axarr[1].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=1)

    fig.savefig(fname + dataType + '_sdcs_bit_' + str(bit) + '_loc_' + loc + '_layer.eps', dpi = 300, bbox_inches='tight', format='eps')


def perLayerBar(sdcsOriginal, sdcsPruned, fname, dataType, loc, bit):
    cmap = plt.get_cmap('tab10')

    x_pos = np.arange(len(sdcsOriginal))

    fig, axarr = plt.subplots(2, figsize=(10,8), sharex=True, sharey=True)
    
    axarr[0].bar(x_pos, sdcsOriginal, color=cmap(0), label=dataType + ' original')
    axarr[0].grid(True, linestyle='dotted')
    axarr[0].set_ylabel('SDC Probability')
    # axarr[0].legend(loc='upper right', frameon=False)

    axarr[1].bar(x_pos, sdcsPruned, color=cmap(1), label=dataType + ' pruned')
    axarr[1].grid(True, linestyle='dotted')
    axarr[1].set_ylabel('SDC Probability')
    # axarr[1].legend(loc='upper right', frameon=False)
    axarr[1].set_xlabel('Layers')
    
    axarr[0].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=1)
    axarr[1].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=1)

    fig.savefig(fname + dataType + '_sdcs_bit_' + str(bit) + '_loc_' + loc + '_layer_bar.eps', dpi = 300, bbox_inches='tight', format='eps')


def perBit(sdcs, fname, layer, loc):
    cmap = plt.get_cmap('tab10')

    fig, axarr = plt.subplots(3, figsize=(10,8), sharex=True)
    # fig.suptitle('SDCs Probabilities by Bit Position')
    dtypes = ['fp32', 'int16', 'int8']

    for idx, dataType in enumerate(dtypes):
        x_pos = np.arange(len(sdcs[idx]))
        axarr[idx].bar(x_pos, sdcs[idx], align='center', color=cmap(idx + 3), label=dataType)
        axarr[idx].legend(loc='upper right', frameon=False)
        axarr[idx].set_ylabel('SDC Probability')


    axarr[2].set_xlabel('Bit Position')
    fig.savefig(fname + 'sdcs_layer_' + str(layer) + '_loc_' + loc + '_bit.eps', dpi = 300, bbox_inches='tight', format='eps')


def perLayerVsAcc(sdcsOriginal, sdcsPruned, acc1sOriginal, accs1sPruned, fname, dataType, loc, bit):
    cmap = plt.get_cmap('tab10')

    x_pos = np.arange(len(sdcsOriginal))

    fig, axarr = plt.subplots(2, figsize=(10,8), sharex=True, sharey=True)
    
    goldenOriginal = acc1sOriginal.pop(0)
    axarr[0].plot(x_pos, sdcsOriginal, 'o-', c=cmap(0), label='SDC')
    axarr[0].bar(x_pos, acc1sOriginal, color=cmap(1), label='Acc@1 Faulty')
    axarr[0].set_ylabel('Original')

    goldenPruned = accs1sPruned.pop(0)
    axarr[1].plot(x_pos, sdcsPruned, 'p-', c=cmap(0), label='SDC')
    axarr[1].bar(x_pos, accs1sPruned, color=cmap(1), label='Acc@1 Faulty')
    axarr[1].set_ylabel('Pruned')
    axarr[1].set_xlabel('Layers')
    
    axarr[0].plot(x_pos, [goldenOriginal] * len(sdcsOriginal), '--', c=cmap(3), label='Acc@1 Golden')
    axarr[1].plot(x_pos, [goldenPruned] * len(sdcsOriginal), '--', c=cmap(3), label='Acc@1 Golden')

    axarr[0].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=3)


    fig.savefig(fname + dataType + '_sdcs_vs_acc_bit_' + str(bit) + '_loc_' + loc + '_layer.eps', dpi = 300, bbox_inches='tight', format='eps')


def goldenFaultyScores(correctGolden, correctFaulty, target, fname, dataType, loc, layer, bit):
    cmap = plt.get_cmap('tab10')

    x_pos = np.arange(len(correctGolden))

    fig, axarr = plt.subplots(2, figsize=(10,8), sharex=True, sharey=True)
    
    axarr[0].scatter(x_pos, correctGolden, s=5, color=cmap(0), alpha=0.5, label=dataType + ' - ' + 'golden')
    axarr[0].plot(x_pos, target, '--', c=cmap(7), label='Correct')
    axarr[0].grid(True, linestyle='dotted')
    axarr[0].set_ylabel('Predicted Label')


    axarr[1].scatter(x_pos, correctFaulty, s=5, color=cmap(3), alpha=0.5, label=dataType + ' - ' + 'faulty')
    axarr[1].plot(x_pos, target, '--', c=cmap(7), label='Correct')
    axarr[1].grid(True, linestyle='dotted')
    axarr[1].set_ylabel('Predicted Label')
    axarr[1].set_xlabel('Inference Round')
    
    axarr[0].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=2)
    axarr[1].legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=2)

    fig.savefig(fname + dataType + '_scores_layer_' + str(layer) + '_bit_' + str(bit) + '_loc_' + loc + '_bar.eps', dpi = 200, bbox_inches='tight', format='eps')


if __name__ == "__main__":
    main()