import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from .finodes import *
from .quantnodes import *
from .bitflip import *
from util.log import *
from util import *


class FI(object):

    def __init__(self, model, fiMode=False, fiBit=None, fiLayer=0, fiFeatures=True, fiWeights=True, 
                quantMode=False, quantType=LinearQuantMode.SYMMETRIC, quantBitFeats=8, quantBitWts=8, 
                quantBitAccum=32, quantClip=False, quantChannel=False, log=False):
        self.model = model
        self.log = log

        self.quantizationMode = quantMode
        self.quantizationType = LinearQuantMode[quantType]
        self.quantizationBitWeights = quantBitWts
        self.quantizationBitParams = quantBitFeats
        self.quantizationBitAccum = quantBitAccum
        self.quantizationClip = quantClip
        self.quantizationChannel = quantChannel
        
        self.injectionMode = fiMode
        self.injectionLayer = fiLayer
        self.injectionBit = fiBit
        self.injectionFeatures = fiFeatures
        self.injectionWeights = fiWeights
        self.numNewLayers = -1

    def traverseModel(self, model):
        for layerName, layerObj in model.named_children():
            newLayer = self.replaceLayer(layerObj, type(layerObj), layerName)
            setattr(model, layerName, newLayer)
            
            if self.has_children(layerObj):
                # For block layers we call recursively
                self.traverseModel(layerObj)

    def replaceLayer(self, layerObj, layerType, layerName):
        if self.quantizationMode:    
            if layerType == nn.Conv2d:
                self.numNewLayers += 1
                return QConv2d(self, self.numNewLayers, layerName, layerObj.weight, layerObj.in_channels, layerObj.out_channels,
                            layerObj.kernel_size, layerObj.stride, layerObj.padding, layerObj.dilation, layerObj.groups, layerObj.bias, 
                            self.quantizationBitWeights, self.quantizationBitParams, self.quantizationBitAccum, self.quantizationType,
                            self.quantizationClip, self.quantizationChannel)
            elif layerType == nn.Linear:
                self.numNewLayers += 1
                return QLinear(self, self.numNewLayers, layerName, layerObj.weight, layerObj.bias, layerObj.in_features, layerObj.out_features, 
                            self.quantizationBitWeights, self.quantizationBitParams, self.quantizationBitAccum, self.quantizationType,
                            self.quantizationClip, self.quantizationChannel)
            else:
                return layerObj

        elif self.injectionMode:
            if layerType == nn.Conv2d:
                self.numNewLayers += 1
                return FIConv2d(self, self.numNewLayers, layerName, layerObj.weight, layerObj.in_channels, layerObj.out_channels,
                            layerObj.kernel_size, layerObj.stride, layerObj.padding, layerObj.dilation, layerObj.groups, layerObj.bias)
            elif layerType == nn.Linear:
                self.numNewLayers += 1
                return FILinear(self, self.numNewLayers, layerName, layerObj.weight, layerObj.bias, layerObj.in_features, layerObj.out_features)
            else:
                return layerObj
        else:
            return layerObj

    def injectFeatures(self, tensorData, tensorShape):
        faulty_res = []

        if len(tensorShape) == 1:
            feature_size = tensorShape
            fault_idx = np.random.randint(0, feature_size)
            
            if self.log:
                logInjectionNode("Node index:", [fault_idx])

            # fault_val = flipFloat(tensorData[fault_idx], bit=self.injectionBit, log=self.log)
            fault_val = bitFlip(tensorData[fault_idx], size=self.quantizationBitParams, 
                        bit=self.injectionBit, log=self.log, quantized=self.quantizationMode)

            faulty_res.append((fault_idx, fault_val))

            return faulty_res

        elif len(tensorShape) == 3:
            batches_size, channels_size, feat_size = tensorShape

            for batch_idx in range(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_idx = np.random.randint(0, feat_size)

                if self.log:
                    logInjectionNode("Node index:", [batch_idx, channel_idx, feat_idx])

                # faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_idx], bit=self.injectionBit, 
                #             log=self.log) 
                faulty_val = bitFlip(tensorData[batch_idx][channel_idx][feat_idx], size=self.quantizationBitParams, 
                            bit=self.injectionBit, log=self.log, quantized=self.quantizationMode) 
        
                faulty_res.append((channel_idx, feat_idx, faulty_val))

            return faulty_res

        if len(tensorShape) == 4:
            batches_size, channels_size, feat_row_size, feat_col_size = tensorShape

            for batch_idx in range(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_row_idx = np.random.randint(0, feat_row_size)
                feat_col_idx = np.random.randint(0, feat_col_size)
                
                if self.log:
                    logInjectionNode("Node index:", [batch_idx, channel_idx, feat_row_idx, feat_col_idx])
                
                # faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_row_idx][feat_col_idx], bit=self.injectionBit, log=self.log) 
                faulty_val = bitFlip(tensorData[batch_idx][channel_idx][feat_row_idx][feat_col_idx], size=self.quantizationBitParams, 
                            bit=self.injectionBit, log=self.log, quantized=self.quantizationMode) 

                faulty_res.append((channel_idx, feat_row_idx, feat_col_idx, faulty_val))
            
            return faulty_res

        else:
            batches_size, feat_size = tensorShape

            for batch_idx in range(0, batches_size):
                feat_idx = np.random.randint(0, feat_size)

                if self.log:
                    logInjectionNode("Node index:", [batch_idx, feat_idx])
 
                # faulty_val = flipFloat(tensorData[batch_idx][feat_idx], bit=self.injectionBit, log=self.log)
                faulty_val = bitFlip(tensorData[batch_idx][feat_idx], size=self.quantizationBitParams, 
                            bit=self.injectionBit, log=self.log, quantized=self.quantizationMode)

                faulty_res.append((feat_idx, faulty_val))

            return faulty_res


    def injectWeights(self, tensorData, tensorShape):
        faulty_res = []

        if tensorShape == 1:
            feature_size = tensorShape
            fault_idx = np.random.randint(0, feature_size)
            
            if self.log:
                logInjectionNode("Node index:", [fault_idx])

            # fault_val = flipFloat(tensorData[fault_idx], bit=self.injectionBit, log=self.log)
            fault_val = bitFlip(tensorData[fault_idx], size=self.quantizationBitWeights, bit=self.injectionBit, log=self.log, quantized=self.quantizationMode)

            faulty_res.append((fault_idx, fault_val))

            return faulty_res

        elif len(tensorShape) == 3:
            filters_size, num_channels, feat_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            channel_idx = np.random.randint(0, num_channels)
            feat_idx = np.random.randint(0, feat_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, channel_idx, feat_idx])

            # faulty_val = flipFloat(tensorData[filter_idx][channel_idx][feat_idx], bit=self.injectionBit, 
            #             log=self.log) 
            faulty_val = bitFlip(tensorData[filter_idx][channel_idx][feat_idx], size=self.quantizationBitWeights, bit=self.injectionBit, 
                        log=self.log, quantized=self.quantizationMode) 
            

            return (filter_idx, channel_idx, feat_idx, faulty_val)

        if len(tensorShape) == 4:
            filters_size, channels_size, feat_row_size, feat_col_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            channel_idx = np.random.randint(0, channels_size)
            feat_row_idx = np.random.randint(0, feat_row_size)
            feat_col_idx = np.random.randint(0, feat_col_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, channel_idx, feat_row_idx, feat_col_idx])

            # faulty_val = flipFloat(tensorData[filter_idx][channel_idx][feat_row_idx][feat_col_idx], 
            #             bit=self.injectionBit, log=self.log) 
            faulty_val = bitFlip(tensorData[filter_idx][channel_idx][feat_row_idx][feat_col_idx], 
                        size=self.quantizationBitWeights, bit=self.injectionBit, log=self.log, quantized=self.quantizationMode) 
         
            return (filter_idx, channel_idx, feat_row_idx, feat_col_idx, faulty_val)

        else:
            filters_size, feat_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            feat_idx = np.random.randint(0, feat_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, feat_idx])

            # faulty_val = flipFloat(tensorData[filter_idx][feat_idx], bit=self.injectionBit, log=self.log)
            faulty_val = bitFlip(tensorData[filter_idx][feat_idx], size=self.quantizationBitWeights, 
                        bit=self.injectionBit, log=self.log, quantized=self.quantizationMode)

            return (filter_idx, feat_idx, faulty_val)

    def setInjectionMode(self, mode):
        if self.log:
            logInjectionWarning("\tSetting injection mode to " + str(mode))
        self.injectionMode = mode

    def setInjectionBit(self, bit):
        if type(bit) == int and bit >= 0 and bit < 32:
            self.injectionBit = bit

    def setInjectionLayer(self, layer):
        self.injectionLayer = layer

    def has_children(self, module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False
