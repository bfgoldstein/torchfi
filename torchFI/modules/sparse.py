import math
import numpy as np
import torch
import torch.nn as nn

from util.quantization import *
from util.log import *


class FIEmbedding(nn.Embedding):
    def __init__(self, fi, name, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(FIEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, 
                                          norm_type, scale_grad_by_freq, sparse, _weight)
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIEmbedding)
        
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
                    logWarning("\tInjecting Fault into feature data of Embedding " + self.name +  " layer.")
                
                indices, faulty_val = self.fi.inject(input.data)
                
                input.data[tuple(indices)] = faulty_val

                return nn.functional.embedding(input, self.weight, self.padding_idx, self.max_norm, 
                                               self.norm_type, self.scale_grad_by_freq, self.sparse)
            else:
                # create new tensor to apply FI
                weightFI = self.weight.clone()

                indices, faulty_val = self.fi.inject(weightFI.data)
                
                weightFI.data[tuple(indices)] = faulty_val                
        
                return nn.functional.embedding(input, weightFI, self.padding_idx, self.max_norm, 
                                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return super(FIEmbedding, self).forward(input)
        
    @staticmethod
    def from_pytorch_impl(fi, name, embedding: nn.Embedding):
        return FIEmbedding(fi, name, embedding.num_embeddings, embedding.embedding_dim, 
                           embedding.padding_idx, embedding.max_norm, embedding.norm_type, 
                           embedding.scale_grad_by_freq, embedding.sparse, embedding.weight)
    
    def __repr__(self):
        return "%s(num_embeddings=%d, embedding_dim=%d, padding_idx=%s, max_norm=%s, " \
                "norm_type=%s, scale_grad_by_freq=%s, sparse=%s, id=%d)" % (
                self.__class__.__name__, 
                self.num_embeddings, 
                self.embedding_dim,
                str(self.padding_idx),
                str(self.max_norm),
                str(self.norm_type),
                str(self.scale_grad_by_freq),
                str(self.sparse),
                self.id)