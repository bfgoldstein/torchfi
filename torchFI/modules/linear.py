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


class FILinear(nn.Linear):

    def __init__(self, fi, name, in_features, out_features, weight=None, bias=None): 
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FILinear)
        
        super(FILinear, self).__init__(in_features, out_features, 
                                       True if bias is not None else False)

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
                # locInjection = np.random.randint(0, 2)
                locInjection = np.random.binomial(1, .5)
            else:
                locInjection = self.fi.injectionFeatures

            if locInjection:             
                if self.fi.log:
                    logWarning("\tInjecting Fault into feature data of Linear "
                                + self.name +  " layer.")
                                
                faulty_res = self.fi.injectFeatures(input.data)
                
                for idx, (indices, faulty_val) in enumerate(faulty_res):
                    # add idx as batch index to indices array
                    input.data[tuple([idx] + indices)] = faulty_val

                return nn.functional.linear(input, self.weight, self.bias)
            else:
                # create new tensor to apply FI
                weightFI = self.weight.clone()

                if self.fi.log:
                    logWarning("\tInjecting Fault into weight data of Linear "
                                + self.name +  " layer.")
                
                indices, faulty_val = self.fi.inject(weightFI.data)
                
                weightFI.data[tuple(indices)] = faulty_val 

                return nn.functional.linear(input, weightFI, self.bias)
        else:
            return super(FILinear, self).forward(input)
    
    @staticmethod
    def from_pytorch_impl(fi, name, linear: nn.Linear):
        return FILinear(fi, name, linear.in_features, linear.out_features, 
                        linear.weight, linear.bias)
    
    def __repr__(self):
        return "%s(in_features=%d, out_features=%d, bias=%s, id=%d)" % (
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                str(True if self.bias is not None else False),
                self.id) 