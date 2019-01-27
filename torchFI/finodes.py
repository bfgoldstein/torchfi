import math
import numpy as np

import torch.nn as nn

from util.log import *

class FIConv2d(nn.Conv2d):
   
    def __init__(self, fi, name, weight, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                dilation=1, groups=1, bias=True):
        super(FIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias)
        
        self.fi = fi
        self.name = name
        self.weight = weight

    def forward(self, input):
        if self.fi.injectionMode:
            # XNOR Operation
            # True only if both injectionFeatures and injectionWeights are True or False
            # False if one of them is True 
            if not(self.fi.injectionFeatures ^ self.fi.injectionWeights):
                # decide where to apply injection
                # weights = 0, activations = 1 
                locInjection = np.random.randint(0, 2)
            else:
                locInjection = self.fi.injectionFeatures

            if locInjection:
                tensorShape = list(input.data.size())
                
                if self.fi.log:
                    logWarning("\tInjecting Fault into feature data of Conv2d " + self.name +  " layer. tensorShape=" + str(tensorShape))
                
                faulty_res = self.fi.injectFeatures(input.data, tensorShape)

                for batch, (channel, feat_row, feat_col, faulty_val) in enumerate(faulty_res):
                    input.data[batch][channel][feat_row][feat_col] = faulty_val

                return nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            else:
                # create new tensor to apply FI
                weightFI = self.weight.clone()

                tensorShape = list(self.weight.data.size())
            
                if self.fi.log:
                    logWarning("\tInjecting Fault into weight data of Conv2d " + self.name +  " layer. tensorShape=" + str(tensorShape))
            
                filter, channel, feat_row, feat_col, faulty_val = self.fi.injectWeights(self.weight.data, tensorShape)

                weightFI.data[filter][channel][feat_row][feat_col] = faulty_val
                #self.weight.data[filter][channel][feat_row][feat_col] = faulty_val
        
                return nn.functional.conv2d(input, weightFI, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
        else:
            return super(FIConv2d, self).forward(input)



class FILinear(nn.Linear):

    def __init__(self, fi, name, weight, bias, in_features, out_features, b=True):
        super(FILinear, self).__init__(in_features, out_features, b)

        self.fi = fi
        self.name = name
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        if self.fi.injectionMode:
            # XNOR Operation
            # True only if both injectionFeatures and injectionWeights are True or False
            # False if one of them is True 
            if not(self.fi.injectionFeatures ^ self.fi.injectionWeights):
                # decide where to apply injection
                # weights = 0, activations = 1 
                locInjection = np.random.randint(0, 2)
            else:
                locInjection = self.fi.injectionFeatures

            if locInjection:
                tensorShape = list(input.data.size())
                
                if self.fi.log:
                    logWarning("\tInjecting Fault into feature data of Liner " + self.name +  " layer. tensorShape=" + str(tensorShape))
                
                faulty_res = self.fi.injectFeatures(input.data, tensorShape)

                if tensorShape == 3:
                    for batch, (channel, feat_idx, faulty_val) in enumerate(faulty_res):
                        input.data[batch][channel][feat_idx] = faulty_val
                else:
                    for batch, (feat_idx, faulty_val) in enumerate(faulty_res):
                        input.data[batch][feat_idx] = faulty_val

                return nn.functional.linear(input, self.weight, self.bias)
            else:
                # create new tensor to apply FI
                weightFI = self.weight.clone()

                tensorShape = list(self.weight.data.size())
                
                if self.fi.log:
                    logWarning("\tInjecting Fault into weight data of Liner " + self.name +  " layer. tensorShape=" + str(tensorShape))
                
                faulty_res = self.fi.injectWeights(self.weight.data, tensorShape)

                if tensorShape == 3:
                    filter, channel, feat_idx, faulty_val = faulty_res
                    weightFI.data[filter][channel][feat_idx] = faulty_val
                else:
                    filter, feat_idx, faulty_val = faulty_res
                    weightFI.data[filter][feat_idx] = faulty_val

                return nn.functional.linear(input, weightFI, self.bias)
        else:
            return super(FILinear, self).forward(input)