###############################################################
# This file was created using part of Distiller project developed by:
#  NervanaSystems https://github.com/NervanaSystems/distiller
# 
# Changes were applied to satisfy torchFI project needs
###############################################################


import math
import numpy as np
from enum import Enum
from collections import OrderedDict

import torch
import torch.nn as nn

from util.log import *


class FIConv2d(nn.Conv2d):
   
    def __init__(self, fi, name, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros', 
                 weight=None, bias=None):
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIConv2d)
        
        super(FIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                       dilation, groups, True if bias is not None else False, padding_mode)
        
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

    def forward(self, input):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            # XNOR Operation
            # True only if both injectionFeatures and injectionWeights are True or False
            # False if one of them is True 
            if not(self.fi.injectionFeatures ^ self.fi.injectionWeights):
                # decide where to apply injection
                # weights = 0, activations = 1 
                #locInjection = np.random.randint(0, 2)
                locInjection = np.random.binomial(1, .5)
            else:
                locInjection = self.fi.injectionFeatures

            if locInjection:
                if self.fi.log:
                    logWarning("\tInjecting Fault into feature data of Conv2d " + self.name +  " layer.")
                
                faulty_res = self.fi.injectFeatures(input.data)
                
                for idx, (indices, faulty_val) in enumerate(faulty_res):
                    # add idx as batch index to indices array
                    input.data[tuple([idx] + indices)] = faulty_val

                # return nn.functional.conv2d(input, self.weight, self.bias, self.stride, 
                #                             self.padding, self.dilation, self.groups)
                return super(FIConv2d, self).conv2d_forward(input, self.weight)
            else:
                # create new tensor to apply FI
                weightFI = self.weight.clone()
            
                if self.fi.log:
                    logWarning("\tInjecting Fault into weight data of Conv2d " + self.name +  " layer.")
            
                indices, faulty_val = self.fi.inject(weightFI.data)
                
                weightFI.data[tuple(indices)] = faulty_val
        
                # return nn.functional.conv2d(input, weightFI, self.bias, self.stride,
                #                             self.padding, self.dilation, self.groups)
                return super(FIConv2d, self).conv2d_forward(input, weightFI)
        else:
            return super(FIConv2d, self).forward(input)

        @staticmethod
        def from_pytorch_impl(fi, name, conv2d: nn.Conv2d):
            return FIConv2d(fi, name, conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, 
                                       conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups, 
                                       conv2d.padding_mode, conv2d.weight, conv2d.bias)

        def __repr__(self):
            return "%s(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d, " \
                    "dilation=%d, groups=%d, padding_mode=%s, bias=%s, id=%d)" % (
                    self.__class__.__name__,
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    self.padding_mode,
                    str(True if self.bias is not None else False),
                    self.id)