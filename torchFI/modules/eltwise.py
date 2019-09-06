###############################################################
# This file was created using part of Distiller project developed by:
#  NervanaSystems https://github.com/NervanaSystems/distiller
# 
# Changes were applied to satisfy torchFI project needs
###############################################################
#
# Copyright (c) 2018 Intel Corporation
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

from operator import setitem
from functools import reduce

import torch
import torch.nn as nn
import distiller.modules as dist


class FIEltwiseAdd(dist.EltwiseAdd):
    def __init__(self, fi, name, inplace=False):
        super(FIEltwiseAdd, self).__init__(inplace)
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIEltwiseAdd)

    def forward(self, *input):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            resFI = input[0].clone()
            
            indices, faulty_val = self.fi.inject(resFI.data)
            
            resFI.data[tuple(indices)] = faulty_val
                        
            if self.inplace:
                for t in input[1:]:
                    resFI += t
            else:
                for t in input[1:]:
                    resFI = resFI + t
            return resFI
        else:
            return super(FIEltwiseAdd, self).forward(*input)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwiseadd: dist.EltwiseAdd):
        return FIEltwiseAdd(fi, name, eltwiseadd.inplace)
    
    def __repr__(self):
        return "%s(name=%s, inplace=%s, id=%d)" % (
                self.__class__.__name__,
                self.name,
                str(self.inplace), 
                self.id)


class FIEltwiseMult(dist.EltwiseMult):
    def __init__(self, fi, name, inplace=False):
        super(FIEltwiseMult, self).__init__(inplace)
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIEltwiseMult)

    def forward(self, *input):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            resFI = input[0].clone()
            
            indices, faulty_val = self.fi.inject(resFI.data)
            
            resFI.data[tuple(indices)] = faulty_val
                    
            if self.inplace:
                for t in input[1:]:
                    resFI *= t
            else:
                for t in input[1:]:
                    resFI = resFI * t
            return resFI
        else:
            return super(FIEltwiseMult, self).forward(*input)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwisemult: dist.EltwiseMult):
        return FIEltwiseMult(fi, name, eltwisemult.inplace)
    
    def __repr__(self):
        return "%s(name=%s, inplace=%s, id=%d)" % (
                self.__class__.__name__,
                self.name,
                str(self.inplace), 
                self.id)
               
    
#TODO: Check for fault injection on Y value
class FIEltwiseDiv(dist.EltwiseDiv):
    def __init__(self, fi, name, inplace=False):
        super(FIEltwiseDiv, self).__init__(inplace)
        self.fi = fi
        self.name = name
        self.id = fi.addNewLayer(name, FIEltwiseDiv)

    def forward(self, x: torch.Tensor, y):
        if self.fi.injectionMode and self.id == self.fi.injectionLayer:
            xFI = x.clone()
            
            indices, faulty_val = self.fi.inject(xFI.data)
            
            xFI.data[tuple(indices)] = faulty_val
            
            if self.inplace:
                return xFI.div_(y)
            return xFI.div(y)
        else:
            return super(FIEltwiseDiv, self).forward(x, y)             

    @staticmethod
    def from_pytorch_impl(fi, name, eltwisediv: dist.EltwiseDiv):
        return FIEltwiseDiv(fi, name, eltwisediv.inplace)
    
    def __repr__(self):
        return "%s(name=%s, inplace=%s, id=%d)" % (
                self.__class__.__name__,
                self.name,
                str(self.inplace), 
                self.id)
