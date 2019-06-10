import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl

import torch

from util import *

parser = argparse.ArgumentParser(description='TorchFI Plot')
parser.add_argument('--original', metavar='DIR', dest='origData',
                    help='path to records')
parser.add_argument('--pruned', metavar='DIR', dest='prunedData', 
                    help='path to pruned records')
parser.add_argument('--prunedComp', metavar='DIR', dest='prunedCompData', 
                    help='path to pruned records')
parser.add_argument('-feats', '--features', dest='features', action='store_true',
                    help='plot data from FI on features/activations')
parser.add_argument('-wts', '--weights', dest='weights', action='store_true',
                    help='plot data from FI on weights')
parser.add_argument('--layers', default=54, dest='nlayers', type=int,
                    help='number of layers to plot')
parser.add_argument('--data_type', default='fp32', dest='dataType', type=str,
                    help='data type to plot')                    
parser.add_argument('--plotLayer', dest='perlayer', action='store_true',
                    help='plot data per Layer')
parser.add_argument('--acc', dest='acc', action='store_true',
                    help='plot sdcs vs acc@1')
parser.add_argument('--niter', dest='niter', type=int,
                    help='number of iterations to read')

def main():
    args = parser.parse_args()
    
    if args.weights:
        loc = 'weights'
    else:
        loc = 'features'

    sdcsOriginalArr = []
    sdcsPrunedArr  = []
    sdcsPrunedCompArr = []

    if args.perlayer:
        dataTypes = ['fp32', 'int16', 'int8']
        # dataTypes = ['fp32']
        for dataType in dataTypes:
            origDataPath = os.path.join(args.origData, dataType)
            prunedDataPath = os.path.join(args.prunedData, dataType)
            prunedCompDataPath = os.path.join(args.prunedCompData, dataType)
            _, sdcsOriginal, acc1sOriginal = loadRecordsLayerAvg(dataType, origDataPath, args.nlayers, 'random', loc, args.niter, 'resnet50_')
            _, sdcsPruned, accs1sPruned = loadRecordsLayerAvg(dataType, prunedDataPath, args.nlayers, 'random', loc, args.niter, 'resnet50_pruned_')
            _, sdcsPrunedComp, accs1sPrunedNew = loadRecordsLayerAvg(dataType, prunedCompDataPath, args.nlayers, 'random', loc, args.niter, 'resnet50_pruned_')
            # perLayer(sdcsOriginal, sdcsPruned, 'resnet50_', dataType, loc, 'random')
            # perLayerSameFig(sdcsOriginal, sdcsPruned, sdcsPrunedComp, 'resnet50_', dataType, loc, 'random')
            # perLayerBar(sdcsOriginal, sdcsPruned, 'resnet50_', dataType, loc, 'random')
            # if args.acc:
            #     perLayerVsAcc(sdcsOriginal, sdcsPruned, acc1sOriginal, accs1sPruned, 'resnet50_', dataType, loc, 'random')
            sdcsOriginalArr.append(sdcsOriginal)
            sdcsPrunedArr.append(sdcsPruned)
            sdcsPrunedCompArr.append(sdcsPrunedComp)
        perLayerSameFigFull(sdcsOriginalArr, sdcsPrunedArr, sdcsPrunedCompArr, 'resnet50_', loc, 'random')


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


def perLayerSameFig(sdcsOriginal, sdcsPruned, sdcsPrunedComp, fname, dataType, loc, bit):
    cmap = plt.get_cmap('tab10')

    x_pos = np.arange(len(sdcsOriginal))

    fig, axarr = plt.subplots(1, figsize=(10,8))
    
    axarr.plot(x_pos, sdcsOriginal, '--', c=cmap(0), label=dataType + ' original')
    axarr.plot(x_pos, sdcsPruned, '--', c=cmap(1), label=dataType + ' pruned (100% faults)')
    axarr.plot(x_pos, sdcsPrunedComp, '--', c=cmap(2), label=dataType + ' pruned (20% faults)')
    axarr.grid(True, linestyle='dotted')
    axarr.set_ylabel('SDC Probability')
    axarr.set_xlabel('Layers')

    # axarr[0].legend(loc='upper right', frameon=False)
    axarr.legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=1)

    fig.savefig(fname + dataType + '_sdcs_bit_' + str(bit) + '_loc_' + loc + '_layer_sameFig.eps', dpi = 300, bbox_inches='tight', format='eps')


def perLayerSameFigFull(sdcsOriginal, sdcsPruned, sdcsPrunedComp, fname, loc, bit):
    cmap = plt.get_cmap('tab10')
    color = 0


    fig, axarr = plt.subplots(1, figsize=(10,8))
    
    for orig, pruned, pronedComp, dataType in zip(sdcsOriginal, sdcsPruned, sdcsPrunedComp, ['fp32', 'int16', 'int8']):
        x_pos = np.arange(len(orig))
        axarr.plot(x_pos, orig, '--', c=cmap(color), label=dataType + ' original')
        axarr.plot(x_pos, pruned, '--', c=cmap(color + 1), label=dataType + ' pruned (100% faults)')
        axarr.plot(x_pos, pronedComp, '--', c=cmap(color + 2), label=dataType + ' pruned (20% faults)')
        color += 3

    axarr.grid(True, linestyle='dotted')
    axarr.set_ylabel('% Error')
    axarr.set_xlabel('Layers')

    # axarr[0].legend(loc='upper right', frameon=False)
    axarr.legend(loc='upper center', bbox_to_anchor=(0., 1.10, 1., .102), ncol=3)

    fig.savefig(fname + 'full_sdcs_bit_' + str(bit) + '_loc_' + loc + '_layer_sameFigFull.eps', dpi = 300, bbox_inches='tight', format='eps')


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



if __name__ == "__main__":
    main()