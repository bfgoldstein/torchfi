import math

import torch.nn as nn
 
from util.log import *

class FIConv2d(nn.Conv2d):
   
    def __init__(self, fi, weight, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                dilation=1, groups=1, bias=True):
        super(FIConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias)

        self.fi = fi
        self.weight = weight

    def forward(self, input):        
        tensorShape = list(input.data.size())
        
        if self.fi.log:
            logWarning("\tInjecting Fault into feature data of Conv2d operation. tensorShape=" + str(tensorShape))
        
        faulty_res = self.fi.inject(input.data, tensorShape)

        for batch, (channel, feat_row, feat_col, faulty_val) in enumerate(faulty_res):
            input.data[batch][channel][feat_row][feat_col] = faulty_val

        tensorShape = list(self.weight.data.size())
    
        if self.fi.log:
            logWarning("\tInjecting Fault into weight data of Conv2d operation. tensorShape=" + str(tensorShape))
    
        faulty_res = self.fi.inject(self.weight.data, tensorShape)

        for batch, (channel, feat_row, feat_col, faulty_val) in enumerate(faulty_res):
            self.weight.data[batch][channel][feat_row][feat_col] = faulty_val

        return super(FIConv2d, self).forward(input)



class FILinear(nn.Linear):

    def __init__(self, fi, weight, bias, in_features, out_features, b=True):
        super(FILinear, self).__init__(in_features, out_features, b)

        self.fi = fi
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        tensorShape = list(input.data.size())
        
        if self.fi.log:
            logWarning("\tInjecting Fault into feature data of Liner operation. tensorShape=" + str(tensorShape))
        
        faulty_res = self.fi.inject(input.data, tensorShape)

        if tensorShape == 3:
            for batch, (channel, feat_idx, faulty_val) in enumerate(faulty_res):
                input.data[batch][channel][feat_idx] = faulty_val
        else:
            for batch, (feat_idx, faulty_val) in enumerate(faulty_res):
                input.data[batch][feat_idx] = faulty_val

        tensorShape = list(self.weight.data.size())
        
        if self.fi.log:
            logWarning("\tInjecting Fault into weight data of Liner operation. tensorShape=" + str(tensorShape))
        
        faulty_res = self.fi.inject(self.weight.data, tensorShape)

        if tensorShape == 3:
            for batch, (channel, feat_idx, faulty_val) in enumerate(faulty_res):
                self.weight.data[batch][channel][feat_idx] = faulty_val
        else:
            for batch, (feat_idx, faulty_val) in enumerate(faulty_res):
                self.weight.data[batch][feat_idx] = faulty_val

        return super(FILinear, self).forward(input)