import torch
import torch.nn as nn

import numpy as np

from finodes import *
from bitflip import *
from util.log import *


class FI():

    def __init__(self, model, layer=0):
        self.model = model
        self.layer = layer
        self.log = True
        self.bit = 2


    def inject(self, tensorData, tensorShape):
        faulty_res = []

        if tensorShape == 1:
            feature_size = tensorShape
            fault_idx = np.random.randint(0, feature_size)
            
            fault_val = flipFloat(tensorData[fault_idx], bit=self.bit, log=self.log)

            faulty_res.append((fault_idx, fault_val))

            return faulty_res

        elif len(tensorShape) == 3:
            batches_size, channels_size, feat_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_idx = np.random.randint(0, feat_size)
                 
                faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_idx], bit=self.bit, log=self.log) 
        
                faulty_res.append((channel_idx, feat_idx, faulty_val))

            return faulty_res

        if len(tensorShape) == 4:
            batches_size, channels_size, feat_row_size, feat_col_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                channel_idx = np.random.randint(0, channels_size)
                feat_row_idx = np.random.randint(0, feat_row_size)
                feat_col_idx = np.random.randint(0, feat_col_size)

                faulty_val = flipFloat(tensorData[batch_idx][channel_idx][feat_row_idx][feat_col_idx], bit=self.bit, log=self.log) 

                faulty_res.append((channel_idx, feat_row_idx, feat_col_idx, faulty_val))
            
            return faulty_res

        else:
            batches_size, feat_size = tensorShape

            for batch_idx in xrange(0, batches_size):
                feat_idx = np.random.randint(0, feat_size)
                 
                faulty_val = flipFloat(tensorData[batch_idx][feat_idx], bit=self.bit, log=self.log)

                faulty_res.append((feat_idx, faulty_val))

            return faulty_res
