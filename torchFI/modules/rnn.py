#
# This file is part of Distiller project and was developed by:
#  NervanaSystems https://github.com/NervanaSystems/distiller
# 
# Minor changes were applied to satisfy torchFI project needs
# 
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

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import distiller.modules as dist

from .eltwise import FIEltwiseAdd, FIEltwiseMult
from .linear import FILinear
from itertools import product

# __all__ = ['FILSTMCell', 'FILSTM', 'convert_model_to_distiller_lstm']


# There is prevalent use of looping that depends on tensor sizes done in this implementation.
# This does not play well with the PyTorch tracing mechanism, and emits several different warnings.
# For "simple" cases, such as SummaryGraph creating a single trace based on a single forward pass,
# this is not an actual problem.
# TODO: Check if/how it's possible to have a tracer-friendly implementation

class FILSTMCell(dist.DistillerLSTMCell):
    """
    A single LSTM block.
    The calculation of the output takes into account the input and the previous output and cell state:
    https://pytorch.org/docs/stable/nn.html#lstmcell
    Args:
        input_size (int): the size of the input
        hidden_size (int): the size of the hidden state / output
        bias (bool): use bias. default: True

    """
    def __init__(self, fi, name, input_size, hidden_size, bias=True):
        self.fi = fi
        self.name = name
        
        super(FILSTMCell, self).__init__(input_size, hidden_size, bias)
        
        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = FILinear(self.fi, 'fc_gate_x', input_size, hidden_size * 4)
        self.fc_gate_h = FILinear(self.fi, 'fc_gate_h', hidden_size, hidden_size * 4)
        self.eltwiseadd_gate = FIEltwiseAdd(self.fi, 'eltwiseadd_gate')
        
        # Calculate cell:
        self.eltwisemult_cell_forget = FIEltwiseMult(self.fi, 'eltwisemult_cell_forget')
        self.eltwisemult_cell_input = FIEltwiseMult(self.fi, 'eltwisemult_cell_input')
        self.eltwiseadd_cell = FIEltwiseAdd(self.fi, 'eltwiseadd_cell')
        # Calculate hidden:
        self.eltwisemult_hidden = FIEltwiseMult(self.fi, 'eltwisemult_hidden')
        self.init_weights()
        
    @staticmethod
    def from_pytorch_impl(fi, name, lstmcell: nn.LSTMCell):
        module = FILSTMCell(fi, name, input_size=lstmcell.input_size, hidden_size=lstmcell.hidden_size, bias=lstmcell.bias)
        
        module.fc_gate_x.weight = nn.Parameter(lstmcell.weight_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(lstmcell.weight_hh.clone().detach())
        
        if lstmcell.bias:
            module.fc_gate_x.bias = nn.Parameter(lstmcell.bias_ih.clone().detach())
            module.fc_gate_h.bias = nn.Parameter(lstmcell.bias_hh.clone().detach())

        return module
    
    def __repr__(self):
        ret =  "%s(input_size=%d, hidden_size=%d, bias=%s)" % (
                self.__class__.__name__, 
                self.input_size, 
                self.hidden_size,
                str(True if self.bias is not None else False))
        # Cell gates
        gates = [self.fc_gate_x,
                 self.fc_gate_h,
                 self.eltwiseadd_gate,
                 self.eltwisemult_cell_forget,
                 self.eltwisemult_cell_input,
                 self.eltwiseadd_cell,
                 self.eltwisemult_hidden]
        for idx, gate in enumerate(gates):
            ret += '\n'
            ret += "\t (" + str(idx) + "): " + gate.__repr__()

        return ret
    
def process_sequence_wise(cell, x, h=None):
    return dist.process_sequence_wise(cell, x)


def _repackage_hidden_unidirectional(h):
    return dist.rnn._repackage_hidden_unidirectional(h)


def _repackage_hidden_bidirectional(h_result):
    return dist.rnn._repackage_hidden_bidirectional(h_result)

def _unpack_bidirectional_input_h(h):
    return dist.rnn._unpack_bidirectional_input_h(h)


class FILSTM(dist.DistillerLSTM):
    """
    A modular implementation of an LSTM module.
    Args:
        input_size (int): size of the input
        hidden_size (int): size of the hidden connections and output.
        num_layers (int): number of LSTMCells
        bias (bool): use bias
        batch_first (bool): the format of the sequence is (batch_size, seq_len, dim). default: False
        dropout : dropout factor
        bidirectional (bool): Whether or not the LSTM is bidirectional. default: False (unidirectional).
        bidirectional_type (int): 1 or 2, corresponds to type 1 and type 2 as per
            https://github.com/pytorch/pytorch/issues/4930. default: 2
    """
    def __init__(self, fi, name, input_size, hidden_size, num_layers, bias=True, batch_first=False,
                 dropout=0.5, bidirectional=False, bidirectional_type=2):
        self.fi = fi
        self.name = name
        super(FILSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, 
                                     dropout, bidirectional, bidirectional_type)

    def _create_cells_list(self, hidden_size_scale=1):
        # We always have the first layer
        cells = nn.ModuleList([FILSTMCell(self.fi, 'cell_1',self.input_size, self.hidden_size, self.bias)])
        for i in range(1, self.num_layers):
            cells.append(FILSTMCell(self.fi, 'cell_' + str(i + 1), hidden_size_scale * self.hidden_size, self.hidden_size, self.bias))
        return cells
    
    @staticmethod
    def from_pytorch_impl(fi, name, lstm: nn.LSTM):
        bidirectional = lstm.bidirectional

        module = FILSTM(fi, name, lstm.input_size, lstm.hidden_size, lstm.num_layers, bias=lstm.bias,
                        batch_first=lstm.batch_first, dropout=lstm.dropout, bidirectional=bidirectional)
        
        param_gates = ['i', 'h']

        param_types = ['weight']
        if lstm.bias:
            param_types.append('bias')

        suffixes = ['']
        if bidirectional:
            suffixes.append('_reverse')

        for i in range(lstm.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = module.cells[i] if psuffix == '' else module.cells_reverse[i]
                lstm_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(lstm, lstm_pth_param_name).clone().detach()  # e.g. `lstm.weight_ih_l0.detach()`
                setattr(gate, ptype, nn.Parameter(param_tensor))

        return module
    
    def __repr__(self):
        ret = "%s(%d, %d, num_layers=%d, dropout=%.2f, bidirectional=%s)" % \
               (self.__class__.__name__,
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout_factor,
                self.bidirectional)
    
        ret += '\n'
        ret += self.cells.__repr__()
        
        return ret
