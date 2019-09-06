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

from util.quantization import *
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

       
class LinearQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3
    
            
class QFILinear(FILinear):
        
    def __init__(self, fi, id, name, in_features, out_features, num_bits_acts=8, num_bits_params=8, 
                 num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC, clip_acts=False, per_channel_wts=False, 
                 weight=None, bias=None):

        super(QFILinear, self).__init__(fi, id, name, in_features, out_features, weight, bias)
        
        self.num_bits_acts = num_bits_acts
        self.num_bits_accum = num_bits_accum
        self.mode = mode
        self.clip_acts = clip_acts

        signed = mode != LinearQuantMode.ASYMMETRIC_UNSIGNED
        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=signed)
        # The accumulator is always signed
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True)

        self.num_bits_params = num_bits_params
        self.per_channel_wts = per_channel_wts

        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            num_bits_params, signed=mode != LinearQuantMode.ASYMMETRIC_UNSIGNED)

        # Quantize weights - overwrite FP32 weights
        w_scale, w_zero_point = _get_tensor_quantization_params(self.weight, num_bits_params, self.mode,
                                                                per_channel=per_channel_wts)

        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_zero_point', w_zero_point)
        linear_quantize_clamp(self.weight.data, self.w_scale, self.w_zero_point, self.params_min_q_val,
                              self.params_max_q_val, inplace=True)

        # Quantize bias
        self.has_bias = hasattr(self, 'bias') and self.bias is not None
        if self.has_bias:
            b_scale, b_zero_point = _get_tensor_quantization_params(self.bias, num_bits_params, self.mode)
            self.register_buffer('b_scale', b_scale)
            self.register_buffer('b_zero_point', b_zero_point)
            base_b_q = linear_quantize_clamp(self.bias.data, self.b_scale, self.b_zero_point,
                                             self.params_min_q_val, self.params_max_q_val)
            # Dynamic ranges - save in auxiliary buffer, requantize each time based on dynamic input scale factor
            self.register_buffer('base_b_q', base_b_q)
        
        self.current_in_scale = 1
        self.current_in_zero_point = 0
        self.current_accum_scale = 1

    def forward(self, inputs):

        in_scales, in_zero_points = self.get_inputs_quantization_params(inputs)
        
        # Quantize inputs
        inputs_q = []

        input_q = linear_quantize_clamp(inputs.data, in_scales, in_zero_points,
                                        self.acts_min_q_val, self.acts_max_q_val, inplace=False)

        inputs_q.append(torch.autograd.Variable(input_q))

        self.current_accum_scale = self.current_in_scale * self.w_scale
        if self.per_channel_wts:
            self.current_accum_scale = self.current_accum_scale.squeeze(dim=-1)

        if self.has_bias:
            # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
            self.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                  self.current_accum_scale / self.b_scale, 0,
                                                                  self.accum_min_q_val, self.accum_max_q_val)

        if self.mode != LinearQuantMode.SYMMETRIC:
            input_q += self.current_in_zero_point
            self.weight.data += self.w_zero_point

        accum = super(QFILinear, self).forward(input_q)

        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        if self.mode != LinearQuantMode.SYMMETRIC:
            self.weight.data -= self.w_zero_point
        

        # Re-quantize accumulator to quantized output range
        out_scale, out_zero_point = self.get_output_quantization_params(accum)
        requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point)
        out_q = linear_quantize_clamp(accum.data, requant_scale, requant_zero_point,
                                      self.acts_min_q_val, self.acts_max_q_val, inplace=True)

        # De-quantize back to FP32
        out_f = linear_dequantize(out_q, out_scale, out_zero_point, inplace=True)

        return torch.autograd.Variable(out_f)

    def get_inputs_quantization_params(self, input):
        self.current_in_scale, self.current_in_zero_point = _get_tensor_quantization_params(input, self.num_bits_acts,
                                                                                            self.mode,clip=self.clip_acts)
        return self.current_in_scale, self.current_in_zero_point

    def get_output_quantization_params(self, accumulator):
        y_f = accumulator / self.current_accum_scale
        return _get_tensor_quantization_params(y_f, self.num_bits_acts, self.mode, clip=self.clip_acts)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return output_scale / self.current_accum_scale, output_zero_point
    
    @staticmethod
    def from_pytorch_impl(fi, id, name, linear: nn.Linear):
        return QFILinear(fi, id, name, linear.in_features, linear.out_features, 
                         fi.quantizationBitWeights, fi.quantizationBitFeatures, 
                         fi.quantizationBitAccum, fi.quantizationType, 
                         fi.quantizationClip, fi.quantizationChannel, 
                         linear.weight, linear.bias)
        
    
def _get_tensor_quantization_params(tensor, num_bits, mode, clip=False, per_channel=False):
    if per_channel and tensor.dim() not in [2, 4]:
        raise ValueError('Per channel quantization possible only with 2D or 4D tensors (linear or conv layer weights)')
    dim = 0 if clip or per_channel else None
    if mode == LinearQuantMode.SYMMETRIC:
        sat_fn = get_tensor_avg_max_abs if clip else get_tensor_max_abs
        sat_val = sat_fn(tensor, dim)
        scale, zp = symmetric_linear_quantization_params(num_bits, sat_val)
    else:   # Asymmetric mode
        sat_fn = get_tensor_avg_min_max if clip else get_tensor_min_max
        sat_min, sat_max = sat_fn(tensor, dim)
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    if per_channel:
        # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
        dims = [scale.shape[0]] + [1] * (tensor.dim() - 1)
        scale = scale.view(dims)
        zp = zp.view(dims)

    return scale, zp