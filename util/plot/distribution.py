import os
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl


def loadScores(deltaGoldenPath, deltaFaultyPath, deltaFigName, deltaPlotTitle):

    def getData(path):
        ret = []
        with open(path, 'r') as scores:
            for line in scores:
                top1 = np.array(line.split(), dtype=np.float32)
                top2 = np.array(next(scores).split(), dtype=np.float32)
                ret.append(np.subtract(top1, top2))
                for _ in range(0, 3):
                    next(scores)
            
            return np.hstack(ret)

    goldenScores = getData(deltaGoldenPath)
    faultyScores = getData(deltaFaultyPath)
    # plotBarDeltaTop12(goldenScores, faultyScores, deltaFigName, deltaPlotTitle)
    plotBarRatio(goldenScores, faultyScores, deltaFigName, deltaPlotTitle)


def plotBarRatio(goldenScores, faultyScores, deltaFigName, deltaPlotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    y = np.arange(0, len(goldenScores))

    plt.barh(y, np.true_divide(faultyScores, goldenScores), align='center', color='b')
    
    ax.set_xlim(0, 15)
    ax.set_xlabel(r'$\Delta$' + ' (golden - faulty)')
    ax.set_ylabel('# Input Sample')
    ax.set_title(r'$\Delta$' + ' score' + '\n ' + deltaPlotTitle)

    fig.tight_layout()
    plt.savefig(deltaFigName + '.png', dpi = 200)
    # plt.show()


def plotBarDeltaTop12(goldenScores, faultyScores, deltaFigName, deltaPlotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    y = np.arange(0, len(goldenScores))

    plt.barh(y, goldenScores, align='center', color='b', label='Correct Prediction')
    
    y = np.arange(y[-1], y[-1] + len(faultyScores))
    plt.barh(y, faultyScores, align='center', color='r', label='Miss Prediction')

    ax.set_xlim(0, 15)
    ax.set_xlabel(r'$\Delta$' + ' (golden - faulty)')
    ax.set_ylabel('# Input Sample')
    ax.set_title(r'$\Delta$' + ' score' + '\n ' + deltaPlotTitle)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


    fig.tight_layout()
    plt.savefig(deltaFigName + '.png', dpi = 200)
    # plt.show()


def loadDeltas(deltaMissPath, deltaCorretPath, deltaFigName, deltaPlotTitle):

    def getData(path):
        with open(path, 'r') as deltas:
            line = deltas.readline()
            line = line.split()
            ret = np.array(line, dtype=np.float32)
            return ret

    deltaCorrect = getData(deltaCorretPath)
    deltaMiss = getData(deltaMissPath)
    plotBarDelta(deltaCorrect, deltaMiss, deltaFigName, deltaPlotTitle)


def plotBarDelta(deltaCorrect, deltaMiss, deltaFigName, deltaPlotTitle):
    mpl.rcParams['figure.dpi'] = 200
  
    fig, ax = plt.subplots()
    
    y = np.arange(0, len(deltaCorrect))

    plt.barh(y, deltaCorrect, align='center', color='b', label='Correct Prediction')
    
    y = np.arange(y[-1], y[-1] + len(deltaMiss))
    plt.barh(y, deltaMiss, align='center', color='r', label='Miss Prediction')

    ax.set_xlim(-8, 8)
    ax.set_xlabel(r'$\Delta$' + ' (golden - faulty)')
    ax.set_ylabel('# Input Sample')
    ax.set_title(r'$\Delta$' + ' score' + '\n ' + deltaPlotTitle)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)


    fig.tight_layout()
    plt.savefig(deltaFigName + '.png', dpi = 200)
    # plt.show()


def main():
    deltaMissPath = sys.argv[1]
    deltaCorretPath = sys.argv[2]
    deltaFigName = sys.argv[3]
    deltaPlotTitle = sys.argv[4]
    
    # loadDeltas(deltaMissPath, deltaCorretPath, deltaFigName, deltaPlotTitle)
    loadScores(deltaMissPath, deltaCorretPath, deltaFigName, deltaPlotTitle)


if __name__ == "__main__":
    main()