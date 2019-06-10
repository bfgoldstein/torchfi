import sys
sys.path.append("/home/bfgoldstein/")
from torchfi.util import *
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab10')

def removeLowerDiagonal_(adj):
    for i in range(adj.shape[0]):
        for j in range(0, i + 1):
            adj[i][j] = 0

            
def drawMissClassNhBar(adj):
    fig, axarr = plt.subplots(1, figsize=(10,10))
    
    arr = []
    for i, row in enumerate(adj):
        arr.append((sum(row), i))
                    
    sort_arr = sorted(arr, key=lambda x: int(x[0]), reverse=True)[:40]
                    
    y_pos = np.arange(len(sort_arr))

    axarr.barh(y_pos, [val[0] for val in sort_arr], align='center', color=cmap(0), alpha=0.5)
    axarr.set_yticks(y_pos)
    
    axarr.set_yticklabels([str(item[1]) for item in sort_arr])
    axarr.invert_yaxis()  # labels read top-to-bottom
    axarr.set_xlabel('total of miss classification')
    axarr.set_title('Amount of miss classification of class N')

    plt.show()

    
def drawMissClassMhBar(adj):
    fig, axarr = plt.subplots(1, figsize=(10,10))

    arr = []
    for i, col in enumerate(np.transpose(adj)):
        arr.append((sum(col), i))
                    
    sort_arr = sorted(arr, key=lambda x: int(x[0]), reverse=True)[:40]
                    
    y_pos = np.arange(len(sort_arr))

    axarr.barh(y_pos, [val[0] for val in sort_arr], align='center', color=cmap(2), alpha=0.5)
    axarr.set_yticks(y_pos)
    
    axarr.set_yticklabels([str(item[1]) for item in sort_arr])
    axarr.invert_yaxis()  # labels read top-to-bottom
    axarr.set_xlabel('total of miss classification')
    axarr.set_title('Amount of miss classification to class M')

    plt.show()
    
    
def drawMissClassN2MhBar(adj):
    fig, axarr = plt.subplots(1, figsize=(10,10))
  
    arr = []
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i][j] > 0:
                    arr.append((adj[i][j], i, j))
                    
    sort_arr = sorted(arr, key=lambda x: int(x[0]), reverse=True)[:40]
                    
    y_pos = np.arange(len(sort_arr))

    axarr.barh(y_pos, [val[0] for val in sort_arr], align='center', color=cmap(3), alpha=0.5)
    axarr.set_yticks(y_pos)
    
    axarr.set_yticklabels([str(item[1]) + " -> " + str(item[2]) for item in sort_arr])
    axarr.invert_yaxis()  # labels read top-to-bottom
    axarr.set_xlabel('total of miss classification')
    axarr.set_title('Amount of miss classification from class N to M')

    plt.show()
    
    
def drawScatter(adj, labels):
    fig, axarr = plt.subplots(1, figsize=(10,10))
    cmap = plt.cm.get_cmap('Reds')
    
    x = []
    y = []
    val = []
    
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i][j] > 0:
                x.append(j)
                y.append(i)
                val.append(adj[i][j])
    
    s = [0.4 * v for v in val]
    sc = axarr.scatter(x, y, c=val, cmap=cmap, s=s)
    axarr.set_xlabel(labels[0])
    axarr.set_ylabel(labels[1])
    axarr.invert_yaxis()
    axarr.set_title(labels[0] + ' vs ' + labels[1])
    plt.colorbar(sc)
    plt.show()


# Load records from faulty and golden runs

mtype = 'original'
dtype = 'fp32'
fname = dtype

# mtype = 'pruned'
# dtype = 'int8'
# fname = 'pruned_int8'

# mtype = 'original'
# dtype = 'int8'
# fname = dtype

path = '/home/bfgoldstein/results_torchfi/randomBit/'
# path = '/home/bfgoldstein/qualify/experiments/'
faulty_suffix = '_layer_0_bit_random_loc_weights_iter_5'
# faulty_suffix = '_layer_1_bit_0_loc_weights_iter_1'

faulty_record = loadRecord(path + mtype + '/' + dtype + '/resnet50_' + fname + faulty_suffix)
golden_record = loadRecord(path + mtype + '/' + dtype + '/resnet50_' + fname + '_golden')


# Create adjacency matrices from faulty, golden and faulty vs golden runs

# adj_faulty = getAdjacency(faulty_record.getTargetLabels(), faulty_record.getTop1PredictionLabels())
# adj_golden = getAdjacency(golden_record.getTargetLabels(), golden_record.getTop1PredictionLabels())

# adj_fg = getAdjacency(golden_record.getTop1PredictionLabels(), faulty_record.getTop1PredictionLabels())

adj_faulty, err_f = getAdjacency(faulty_record.getTargetLabels(), faulty_record.getTop1PredictionLabelsBatch())
adj_golden, err_g = getAdjacency(golden_record.getTargetLabels(), golden_record.getTop1PredictionLabelsBatch())

adj_gf, _ = getAdjacency(golden_record.getTop1PredictionLabelsBatch(), faulty_record.getTop1PredictionLabelsBatch())


print("NaN errors")
print("Faulty %d, Golden %d" % (err_f, err_g))


# Remove lower matrix and diagonal
removeLowerDiagonal_(adj_faulty)
removeLowerDiagonal_(adj_golden)
removeLowerDiagonal_(adj_gf)


drawScatter(adj_golden, ['Golden', 'Correct'])
drawScatter(adj_faulty, ['Faulty', 'Correct'])
drawScatter(adj_gf, ['Golden', 'Faulty'])


temp = 0
total = 0
ss = 0
for i, r in enumerate(adj_gf):
    total += sum(r)
    if max(r) > temp:
        temp = max(r)
        row = i
        
print(row, temp, sum(adj_gf[row]), total)

drawMissClassNhBar(adj_golden)
drawMissClassNhBar(adj_faulty)
drawMissClassNhBar(adj_gf)

drawMissClassMhBar(adj_golden)
drawMissClassMhBar(adj_faulty)
drawMissClassMhBar(adj_gf)

drawMissClassN2MhBar(adj_golden)
drawMissClassN2MhBar(adj_faulty)
drawMissClassN2MhBar(adj_gf)

