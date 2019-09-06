###############################################################
# This file was created using part of Distiller project developed by:
#  NervanaSystems https://github.com/NervanaSystems/distiller
# 
# Changes were applied to satisfy torchFI project needs
###############################################################
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch.nn as nn
import distiller.modules as dist

from util.log import *


class FIMatmul(dist.Matmul):
    """
    A wrapper module for matmul operation between 2 tensors.
    """
    def __init__(self, fi, name):
        super(FIMatmul, self).__init__()
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIMatmul)
        
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            aFI = a.clone()

            if self.fi.log:
                    logWarning("\tInjecting Fault into feature data of Embedding "
                                + self.name +  " layer.")
                    
            indices, faulty_val = self.fi.inject(aFI.data)
            
            aFI.data[tuple(indices)] = faulty_val
            
            return aFI.matmul(b)
        else:  
            return super(FIMatmul, self).forward(a, b)

    @staticmethod
    def from_pytorch_impl(fi, name, matmul: dist.Matmul):
        return FIMatmul(fi, name)
    
    def __repr__(self):
        return "%s(name=%s, id=%d)" % (
                self.__class__.__name__,
                self.name,
                self.id)


class FIBatchMatmul(dist.BatchMatmul):
    """
    A wrapper module for matmul operation between 2 tensors.
    """
    def __init__(self, fi, name):
        super(FIBatchMatmul, self).__init__()
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIBatchMatmul)
        
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            aFI = a.clone()

            indices, faulty_val = self.fi.inject(aFI.data)
            
            aFI.data[tuple(indices)] = faulty_val
            
            return torch.bmm(aFI, b)
        else:  
            return super(FIBatchMatmul, self).forward(a, b)
        
    @staticmethod
    def from_pytorch_impl(fi, name, batchmatmul: dist.BatchMatmul):
        return FIBatchMatmul(fi, name)
    
    def __repr__(self):
        return "%s(name=%s, id=%d)" % (
                self.__class__.__name__,
                self.name,
                self.id)