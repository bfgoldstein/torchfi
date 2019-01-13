import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from finodes import *
from bitflip import *
from util.log import *


class FI(object):

    def __init__(self, model, mode=False, bit=None, log=False, layer=0):
        self.model = model
        self.layer = layer
        self.bit = bit
        self.log = log
        self.injectionMode = mode

    def createFaultyLayer(self):
        layerName, layerObj = self.model._modules.items()[self.layer]

        # Conv2d Object
        if isinstance(layerObj, torch.nn.modules.conv.Conv2d):
            faultyLayer = FIConv2d(self, layerName, layerObj.weight, layerObj.in_channels, layerObj.out_channels,
                                layerObj.kernel_size, layerObj.stride, layerObj.padding, layerObj.dilation,
                                layerObj.groups, layerObj.bias)
            return layerName, faultyLayer
        
        # Linear Object
        elif isinstance(layerObj, torch.nn.modules.Linear):
            faultyLayer = FILinear(self, layerName, layerObj.weight, layerObj.bias, layerObj.in_features, 
                                layerObj.out_features)
            return layerName, faultyLayer
        
        # Sequential Object
        elif isinstance(layerObj, torch.nn.modules.Sequential):
            blocks = {}

            # Bottleneck Object
            randIdxBlockFault = np.random.randint(0, len(layerObj))
            blockObj = layerObj[randIdxBlockFault]
            
            blocks[randIdxBlockFault] = []

            # Layers inside Bottleneck
            randIdxBlockLayerFault = np.random.randint(0, len(blockObj._modules.items()))
            blockLayerName, blockLayerObj = blockObj._modules.items()[randIdxBlockLayerFault]

            while not (isinstance(blockLayerObj, torch.nn.modules.conv.Conv2d) or (blockLayerName == "downsample" and isinstance(blockLayerObj, torch.nn.modules.Sequential))):
                randIdxBlockLayerFault = np.random.randint(0, len(blockObj._modules.items()))
                blockLayerName, blockLayerObj = blockObj._modules.items()[randIdxBlockLayerFault]

            if isinstance(blockLayerObj, torch.nn.modules.conv.Conv2d):
                faultyLayer = FIConv2d(self, layerName + "_Bottleneck_" + str(randIdxBlockFault) + "_" + blockLayerName, blockLayerObj.weight, blockLayerObj.in_channels, 
                                    blockLayerObj.out_channels, blockLayerObj.kernel_size, blockLayerObj.stride, blockLayerObj.padding, 
                                    blockLayerObj.dilation, blockLayerObj.groups, blockLayerObj.bias)
                blocks[randIdxBlockFault].append((blockLayerName, faultyLayer))

            # Downsample Sequential Object
            if blockLayerName == "downsample" and isinstance(blockLayerObj, torch.nn.modules.Sequential):
                downsampleLayers = []
                for downsampleName, downsampleObj in blockLayerObj._modules.items():
                    if isinstance(downsampleObj, torch.nn.modules.conv.Conv2d):
                        faultyLayer = FIConv2d(self, layerName + "_Bottleneck_" + str(randIdxBlockFault) + "_" + blockLayerName + "_" + downsampleName, downsampleObj.weight, 
                                            downsampleObj.in_channels, downsampleObj.out_channels, downsampleObj.kernel_size, downsampleObj.stride,
                                            downsampleObj.padding, downsampleObj.dilation, downsampleObj.groups, downsampleObj.bias)
                        downsampleLayers.append((downsampleName, faultyLayer))
                blocks[randIdxBlockFault].append((blockLayerName, downsampleLayers))
            
            return layerName, blocks
        else:
             raise Exception('Not Implemented Faulty Node')
             
        


    def injectFeatures(self, tensorData, tensorShape):
        faulty_res = []

        if len(tensorShape) == 1:
            feature_size = tensorShape
            fault_idx = np.random.randint(0, feature_size)
            
            if self.log:
                logInjectionNode("Node index:", [fault_idx])

            fault_val = flipFloat(tensorData[fault_idx], bit=self.bit, log=self.log)

            faulty_res.append((fault_idx, fault_val))

            return faulty_res

        elif len(tensorShape) == 3:
            batches_size, channels_size, feat_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_idx = np.random.randint(0, feat_size)

                if self.log:
                    logInjectionNode("Node index:", [batch_idx, channel_idx, feat_idx])

                faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_idx], bit=self.bit, 
                            log=self.log) 
        
                faulty_res.append((channel_idx, feat_idx, faulty_val))

            return faulty_res

        if len(tensorShape) == 4:
            batches_size, channels_size, feat_row_size, feat_col_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_row_idx = np.random.randint(0, feat_row_size)
                feat_col_idx = np.random.randint(0, feat_col_size)
                
                if self.log:
                    logInjectionNode("Node index:", [batch_idx, channel_idx, feat_row_idx, feat_col_idx])
                
                faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_row_idx][feat_col_idx], bit=self.bit, log=self.log) 

                faulty_res.append((channel_idx, feat_row_idx, feat_col_idx, faulty_val))
            
            return faulty_res

        else:
            batches_size, feat_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                feat_idx = np.random.randint(0, feat_size)

                if self.log:
                    logInjectionNode("Node index:", [batch_idx, feat_idx])
 
                faulty_val = flipFloat(tensorData[batch_idx][feat_idx], bit=self.bit, log=self.log)

                faulty_res.append((feat_idx, faulty_val))

            return faulty_res


    def injectWeights(self, tensorData, tensorShape):
        faulty_res = []

        if tensorShape == 1:
            feature_size = tensorShape
            fault_idx = np.random.randint(0, feature_size)
            
            if self.log:
                logInjectionNode("Node index:", [fault_idx])

            fault_val = flipFloat(tensorData[fault_idx], bit=self.bit, log=self.log)

            faulty_res.append((fault_idx, fault_val))

            return faulty_res

        elif len(tensorShape) == 3:
            filters_size, num_channels, feat_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            channel_idx = np.random.randint(0, num_channels)
            feat_idx = np.random.randint(0, feat_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, channel_idx, feat_idx])

            faulty_val = flipFloat(tensorData[filter_idx][channel_idx][feat_idx], bit=self.bit, 
                        log=self.log) 

            return (filter_idx, channel_idx, feat_idx, faulty_val)

        if len(tensorShape) == 4:
            filters_size, channels_size, feat_row_size, feat_col_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            channel_idx = np.random.randint(0, channels_size)
            feat_row_idx = np.random.randint(0, feat_row_size)
            feat_col_idx = np.random.randint(0, feat_col_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, channel_idx, feat_row_idx, feat_col_idx])

            faulty_val = flipFloat(tensorData[filter_idx][channel_idx][feat_row_idx][feat_col_idx], 
                        bit=self.bit, log=self.log) 
         
            return (filter_idx, channel_idx, feat_row_idx, feat_col_idx, faulty_val)

        else:
            filters_size, feat_size = tensorShape

            filter_idx = np.random.randint(0, filters_size)
            feat_idx = np.random.randint(0, feat_size)

            if self.log:
                logInjectionNode("Node index:", [filter_idx, feat_idx])

            faulty_val = flipFloat(tensorData[filter_idx][feat_idx], bit=self.bit, log=self.log)

            return (filter_idx, feat_idx, faulty_val)


    def setinjectionMode(self, mode):
        if self.log:
            logInjectionWarning("\tSetting injection mode to " + str(mode))
        self.injectionMode = mode


    def setInjectionBit(self, bit):
        if type(bit) == int and bit >= 0 and bit < 32:
            self.bit = bit


    def setInjectionLayer(self, layer):
        self.layer = layer