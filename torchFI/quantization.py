from distiller.quantization import *
from distiller.quantization.range_linear import _enum_to_str
import torchFI.modules as tfmods

    
class FIPostTrainLinearQuantizer(PostTrainLinearQuantizer):
    
    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 per_channel_wts=False, model_activation_stats=None, fp16=False, clip_n_stds=None,
                 scale_approx_mult_bits=None):
        super(FIPostTrainLinearQuantizer, self).__init__(model, bits_activations, bits_parameters, bits_accum,
                                                       overrides, mode, clip_acts, per_channel_wts, 
                                                       model_activation_stats, fp16, clip_n_stds, 
                                                       scale_approx_mult_bits)
        
        mode = verify_quant_mode(mode)
        clip_acts = verify_clip_mode(clip_acts)
        if clip_acts == ClipMode.N_STD and clip_n_stds is None:
            raise ValueError('clip_n_stds must not be None when clip_acts set to N_STD')

        if model_activation_stats is not None:
            if isinstance(model_activation_stats, str):
                if not os.path.isfile(model_activation_stats):
                    raise ValueError("Model activation stats file not found at: " + model_activation_stats)
                msglogger.info('Loading activation stats from: ' + model_activation_stats)
                with open(model_activation_stats, 'r') as stream:
                    model_activation_stats = distiller.utils.yaml_ordered_load(stream)
            elif not isinstance(model_activation_stats, (dict, OrderedDict)):
                raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')
        else:
            msglogger.warning("\nWARNING:\nNo stats file passed - Dynamic quantization will be used\n"
                              "At the moment, this mode isn't as fully featured as stats-based quantization, and "
                              "the accuracy results obtained are likely not as representative of real-world results."
                              "\nSpecifically:\n"
                              "  * Not all modules types are supported in this mode. Unsupported modules will remain "
                              "in FP32.\n"
                              "  * Optimizations for quantization of layers followed by Relu/Tanh/Sigmoid are only "
                              "supported when statistics are used.\nEND WARNING\n")

        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters,
                                                    'bits_accum': bits_accum,
                                                    'mode': str(mode).split('.')[1],
                                                    'clip_acts': _enum_to_str(clip_acts),
                                                    'clip_n_stds': clip_n_stds,
                                                    'per_channel_wts': per_channel_wts,
                                                    'fp16': fp16,
                                                    'scale_approx_mult_bits': scale_approx_mult_bits}}
        
        def replace_param_layer(module, name, qbits_map, per_channel_wts=per_channel_wts,
                                mode=mode, fp16=fp16, scale_approx_mult_bits=scale_approx_mult_bits,
                                clip_acts=clip_acts, clip_n_stds=clip_n_stds):
            if fp16:
                return FP16Wrapper(module)
            norm_name = distiller.utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts)
            return RangeLinearQuantParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                                     num_bits_accum=self.bits_accum, mode=mode, clip_acts=clip_acts,
                                                     per_channel_wts=per_channel_wts,
                                                     activation_stats=self.model_activation_stats.get(norm_name, None),
                                                     clip_n_stds=clip_n_stds,
                                                     scale_approx_mult_bits=scale_approx_mult_bits)

        def replace_non_param_layer(wrapper_type, module, name, qbits_map, fp16=fp16,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    clip_acts=clip_acts, clip_n_stds=clip_n_stds):
            if fp16:
                return FP16Wrapper(module)
            norm_name = distiller.utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts)
            try:
                return wrapper_type(module, qbits_map[name].acts, mode=mode, clip_acts=clip_acts,
                                    activation_stats=self.model_activation_stats.get(norm_name, None),
                                    clip_n_stds=clip_n_stds, scale_approx_mult_bits=scale_approx_mult_bits)
            except NoStatsError:
                msglogger.warning('WARNING: {0} - quantization of {1} without stats not supported. '
                                  'Keeping the original FP32 module'.format(name, module.__class__.__name__))
                return module

        def replace_embedding(module, name, qbits_map, fp16=fp16):
            if fp16:
                return FP16Wrapper(module, convert_input=False)
            norm_name = distiller.utils.normalize_module_name(name)
            return RangeLinearEmbeddingWrapper(module, qbits_map[name].wts, mode=mode,
                                                        stats=self.model_activation_stats.get(norm_name, None))

        self.clip_acts = clip_acts
        self.clip_n_stds = clip_n_stds
        self.model_activation_stats = model_activation_stats or {}
        self.bits_accum = bits_accum
        self.mode = mode
        
        self.replacement_factory[tfmods.FIConv2d] = replace_param_layer
        # self.replacement_factory[nn.Conv3d] = replace_param_layer
        self.replacement_factory[tfmods.FILinear] = replace_param_layer

        factory_concat = partial(
            replace_non_param_layer, RangeLinearQuantConcatWrapper)
        factory_eltwiseadd = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseAddWrapper)
        factory_eltwisemult = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseMultWrapper)
        factory_matmul = partial(
            replace_non_param_layer, RangeLinearQuantMatmulWrapper)

        update_wrapper(factory_concat, replace_non_param_layer)
        update_wrapper(factory_eltwiseadd, replace_non_param_layer)
        update_wrapper(factory_eltwisemult, replace_non_param_layer)
        update_wrapper(factory_matmul, replace_non_param_layer)

        self.replacement_factory[distiller.modules.Concat] = factory_concat
        self.replacement_factory[tfmods.FIEltwiseAdd] = factory_eltwiseadd
        self.replacement_factory[tfmods.FIEltwiseMult] = factory_eltwisemult
        self.replacement_factory[tfmods.FIMatmul] = factory_matmul
        self.replacement_factory[tfmods.FIBatchMatmul] = factory_matmul
        self.replacement_factory[tfmods.FIEmbedding] = replace_embedding