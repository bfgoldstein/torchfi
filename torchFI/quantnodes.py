from finodes import *
from quant_util import *

import torch.nn as nn

from enum import Enum
from collections import OrderedDict


class LinearQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3


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


class QConv2d(FIConv2d):
    
    def __init__(self, fi, name, weight, in_channels, out_channels, kernel_size, num_bits_acts, num_bits_params, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits_accum=32, 
                mode=LinearQuantMode.SYMMETRIC, clip_acts=False, per_channel_wts=True):

        super(QConv2d, self).__init__(fi, name, weight, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        
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
        
    def forward(self, input):

        in_scales = []
        in_zero_points = []
        accum_scales = []

        # Quantize inputs
        for batch in input:
            in_scale, in_zero_point = self.get_inputs_quantization_params(batch)

            in_scales.append(in_scale)
            in_zero_points.append(in_zero_point)

            linear_quantize_clamp(batch.data, in_scale, in_zero_point,
                                    self.acts_min_q_val, self.acts_max_q_val, inplace=True)
            
            accum_scale = in_scale * self.w_scale
            
            if self.per_channel_wts:
                accum_scale = accum_scale.squeeze(dim=-1)

            accum_scales.append(accum_scale)

        # TODO: Check accum_scale to work with batch approach
        if self.has_bias:
            # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
            self.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                accum_scale / self.b_scale, 0,
                                                                self.accum_min_q_val, self.accum_max_q_val)

        # Note the main terms within the summation is:
        #   (x_q + zp_x) * (w_q + zp_w)
        # In a performance-optimized solution, we would expand the parentheses and perform the computation similar
        # to what is described here:
        #   https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
        # However, for now we're more concerned with simplicity rather than speed. So we'll just add the zero points
        # to the input and weights and pass those to the wrapped model. Functionally, since at this point we're
        # dealing solely with integer values, the results are the same either way.

        if self.mode != LinearQuantMode.SYMMETRIC:
            for batch, in_zero_point in zip(input, in_zero_points):
                batch += in_zero_point
            self.weight.data += self.w_zero_point

        self.fi.injectionMode = False
        accum = super(QConv2d, self).forward(input)

        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        if self.mode != LinearQuantMode.SYMMETRIC:
            self.weight.data -= self.w_zero_point

        # Re-quantize accumulator to quantized output range
        for batch, accum_scale in zip(accum, accum_scales):
            out_scale, out_zero_point = self.get_output_quantization_params(accum, accum_scale)
            requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point, accum_scale)
            linear_quantize_clamp(batch.data, requant_scale, requant_zero_point,
                                self.acts_min_q_val, self.acts_max_q_val, inplace=True)

            # De-quantize back to FP32
            linear_dequantize(batch.data, out_scale, out_zero_point, inplace=True)

        return torch.autograd.Variable(accum)

    def get_inputs_quantization_params(self, input):
        in_scale, in_zero_point = _get_tensor_quantization_params(input, self.num_bits_acts, self.mode, clip=self.clip_acts)
        return in_scale, in_zero_point

    def get_output_quantization_params(self, accumulator, accum_scale):
        y_f = accumulator / accum_scale
        return _get_tensor_quantization_params(y_f, self.num_bits_acts, self.mode, clip=self.clip_acts)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point, accum_scale):
        return output_scale / accum_scale, output_zero_point


class QLinear(FILinear):
    
    def __init__(self, fi, name, weight, bias, in_features, out_features, num_bits_acts, num_bits_params, 
                b=True, num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC, clip_acts=False, per_channel_wts=True):

        super(QLinear, self).__init__(fi, name, weight, bias, in_features, out_features, b)
        
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
        
    def forward(self, input):

        in_scales = []
        in_zero_points = []
        accum_scales = []

        # Quantize inputs
        for batch in input:
            in_scale, in_zero_point = self.get_inputs_quantization_params(batch)

            in_scales.append(in_scale)
            in_zero_points.append(in_zero_point)

            linear_quantize_clamp(batch.data, in_scale, in_zero_point,
                                    self.acts_min_q_val, self.acts_max_q_val, inplace=True)
            
            accum_scale = in_scale * self.w_scale
            
            if self.per_channel_wts:
                accum_scale = accum_scale.squeeze(dim=-1)

            accum_scales.append(accum_scale)

        # TODO: Check accum_scale to work with batch approach
        if self.has_bias:
            # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
            self.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                accum_scale / self.b_scale, 0,
                                                                self.accum_min_q_val, self.accum_max_q_val)

        # Note the main terms within the summation is:
        #   (x_q + zp_x) * (w_q + zp_w)
        # In a performance-optimized solution, we would expand the parentheses and perform the computation similar
        # to what is described here:
        #   https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
        # However, for now we're more concerned with simplicity rather than speed. So we'll just add the zero points
        # to the input and weights and pass those to the wrapped model. Functionally, since at this point we're
        # dealing solely with integer values, the results are the same either way.

        if self.mode != LinearQuantMode.SYMMETRIC:
            for batch, in_zero_point in zip(input, in_zero_points):
                batch += in_zero_point
            self.weight.data += self.w_zero_point

        self.fi.injectionMode = False
        accum = super(QLinear, self).forward(input)

        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        if self.mode != LinearQuantMode.SYMMETRIC:
            self.weight.data -= self.w_zero_point

        # Re-quantize accumulator to quantized output range
        for batch, accum_scale in zip(accum, accum_scales):
            out_scale, out_zero_point = self.get_output_quantization_params(accum, accum_scale)
            requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point, accum_scale)
            linear_quantize_clamp(batch.data, requant_scale, requant_zero_point,
                                self.acts_min_q_val, self.acts_max_q_val, inplace=True)

            # De-quantize back to FP32
            linear_dequantize(batch.data, out_scale, out_zero_point, inplace=True)

        return torch.autograd.Variable(accum)

    def get_inputs_quantization_params(self, input):
        in_scale, in_zero_point = _get_tensor_quantization_params(input, self.num_bits_acts, self.mode, clip=self.clip_acts)
        return in_scale, in_zero_point

    def get_output_quantization_params(self, accumulator, accum_scale):
        y_f = accumulator / accum_scale
        return _get_tensor_quantization_params(y_f, self.num_bits_acts, self.mode, clip=self.clip_acts)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point, accum_scale):
        return output_scale / accum_scale, output_zero_point