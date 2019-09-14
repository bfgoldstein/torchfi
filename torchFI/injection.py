import torch
import torch.nn as nn
import torchvision.models as models

from functools import reduce
from operator import getitem
import numpy as np

import distiller.modules as dist

from .modules import *
from .bitflip import *
from util import *


class FI(object):

    def __init__(self, model, record=None, fiMode=False, fiBit=None, fiepoch=None, 
                 fiLayer=0, fiFeatures=True, fiWeights=True, log=False):
        self.model = model
        self.record = record
        self.log = log
        
        self.injectionMode = fiMode
        self.injectionLayer = fiLayer
        self.injectionBit = fiBit
        self.injectionFeatures = fiFeatures
        self.injectionWeights = fiWeights
        self.numNewLayers = -1
        self.epoch = fiepoch

        self.numInjections = 0
        
        self.factory = {}
        self.fillFacotry()
        self.layersIds = []
        
    def traverseModel(self, model):
        for layerName, layerObj in model.named_children():
            newLayer = self.replaceLayer(layerObj, type(layerObj), layerName)
            setattr(model, layerName, newLayer)
            
            # For block layers we call recursively
            if self.has_children(layerObj):
                self.traverseModel(layerObj)

    def replaceLayer(self, layerObj, layerType, layerName):
        if self.injectionMode:
            if layerType in self.factory:
                return self.factory[layerType].from_pytorch_impl(self, layerName, layerObj)
        return layerObj

    def injectFeatures(self, tensorData, batchSize):
        faulty_res = []
        
        for batch_idx in range(0, batchSize):
            indices, faulty_val = self.inject(tensorData)
            faulty_res.append(([batch_idx] + indices, faulty_val))
        
        return faulty_res           

    def injectWeights(self, tensorData, tensorShape):
        return self.inject(tensorData)
    
    def inject(self, data: torch.Tensor):

        indices, data_val = getDataFromRandomIndex(data)
        
        # while data_val == 0.0:
        #     indices, data_val = getDataFromRandomIndex(data)

        if self.log:
            logInjectionNode("Node index:", indices)

        faulty_val, bit = bitFlip(data_val, 
                                  size=(self.quantizationBitWeights if self.injectionWeights else self.quantizationBitActivations), 
                                  bit=self.injectionBit, log=self.log, quantized=self.quantizationMode) 
        
        self.injectionMode = False
        self.numInjections += 1
        self.recordData(bit, data_val, faulty_val, indices)

        return indices, faulty_val
    
    def setInjectionMode(self, mode):
        if self.log:
            logInjectionWarning("\tSetting injection mode to " + str(mode))
        self.injectionMode = mode

    def setInjectionBit(self, bit):
        if type(bit) == int and bit >= 0 and bit < 32:
            self.injectionBit = bit

    def setInjectionLayer(self, layer):
        self.injectionLayer = layer

    def setQuantParams(self, args):
        self.quantizationMode = True
        self.quantizationType = args.quant_mode
        self.quantizationBitActivations = args.quant_bacts
        self.quantizationBitWeights = args.quant_bwts
        self.quantizationBitAccum = args.quant_baccum
        
    def has_children(self, module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False
    
    def addNewLayer(self, layerName, layerType):
        self.numNewLayers += 1
        self.layersIds.append((layerName, layerType))
        return self.numNewLayers
    
    def recordData(self, bit, original, faulty, location):
        # pass
        self.record.addFiBit(bit)
        self.record.addOriginalValue(float(original.cpu()))
        self.record.addFiValue(faulty)
        self.record.addFiLocation(location)

    def fillFacotry(self):
        self.factory[nn.Conv2d] = FIConv2d
        self.factory[nn.Linear] = FILinear
        self.factory[nn.LSTM] = FILSTM
        self.factory[nn.LSTMCell] = FILSTMCell
        self.factory[nn.Embedding] = FIEmbedding
        self.factory[dist.EltwiseAdd] = FIEltwiseAdd
        self.factory[dist.EltwiseMult] = FIEltwiseMult
        self.factory[dist.EltwiseDiv] = FIEltwiseDiv
        self.factory[dist.Matmul] = FIMatmul
        self.factory[dist.BatchMatmul] = FIBatchMatmul
        

def getDataFromRandomIndex(data: torch.Tensor):
    # get item from multidimensional list at indices position 
    indices = [getRandomIndex(dim_size) for dim_size in data.shape]
    fiData = reduce(getitem, indices, data)
    return indices, fiData
    
    
def getRandomIndex(max: int):
    return np.random.randint(0, max)