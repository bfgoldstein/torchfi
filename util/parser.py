import argparse
import torchvision.models as models
from distiller.quantization.range_linear import *


def getModelNames():
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    return model_names

def getParser():
        
    #####
    ##  Main Arguments
    #####
    
    parser = argparse.ArgumentParser(description='PyTorch arguments')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=getModelNames(),
                        help='torchvision model architectures: ' +
                            ' | '.join(getModelNames()) +
                            ' (default: resnet18)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for injection')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://1.1.1.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('-l', '--log', dest='log', action='store_true',
                        help='turn loging on')
    parser.add_argument('--log-path', dest='log_path', default=None, type=str,
                        help='path to log folder')
    parser.add_argument('--log-prefix', dest='log_prefix', default=None, type=str,
                        help='prefix of log output folder')
    
    #####
    ##  Training Arguments
    #####
    
    train_group = parser.add_argument_group('Arguments for model training')
    
    train_group.add_argument('--epochs', default=90, type=int, metavar='N',
                             help='number of total epochs to run')
    train_group.add_argument('-lr', '--learning-rate', dest='lr', default=0.1, type=float,
                             metavar='LR', help='initial learning rate')
    train_group.add_argument('-mm', '--momentum', dest='momentum', default=0.9, type=float, metavar='M',
                             help='momentum')
    train_group.add_argument('-gm', '--gamma', dest='gamma', default=0.1, type=float, metavar='M',
                             help='gamma')
    train_group.add_argument('-wd', '--weight-decay', dest='weight_decay', default=1e-4, type=float,
                             metavar='W', help='weight decay (default: 1e-4)')
    train_group.add_argument('--resume', default='', type=str, metavar='PATH',
                             help='path to latest checkpoint (default: none)')
    train_group.add_argument('-tb', '--test-batch-size', dest='test_batch_size', type=int, default=1000, metavar='N',
                             help='input batch size for testing (default: 1000)')
    train_group.add_argument('--save-model', dest='save_model', default=None, type=str,
                             help='save the current model with specified file name')
    train_group.add_argument('--plot', dest='plot', default=None, type=str, 
                             help='full path with file name where plot with loss info will be stored')


    #####
    ##  Fault Injection Arguments
    #####
    
    injection_group = parser.add_argument_group('Arguments for fault injection control')
    
    injection_group.add_argument('--golden', dest='golden', action='store_true', 
                                 help='Run golden version')
    injection_group.add_argument('--faulty', dest='faulty', action='store_true',
                                 help='Run faulty version')
    injection_group.add_argument('-i', '--injection', dest='injection', action='store_true',
                                 help='apply FI model')
    injection_group.add_argument('--layer', default=0, type=int,
                                 help='Layer to inject fault.')
    injection_group.add_argument('--bit', default=None, type=int,
                                 help='Bit to inject fault. MSB=0 and LSB=31')
    injection_group.add_argument('--fiEpoch', default=None, type=int,
                                 help='Epoch to inject fault.')    
    injection_group.add_argument('-feats', '--features', dest='fiFeats', action='store_true',
                                 help='inject FI on features/activations')
    injection_group.add_argument('-wts', '--weights', dest='fiWeights', action='store_true',
                                 help='inject FI on weights')
    injection_group.add_argument('--scores', dest='scores', action='store_true',
                                 help='turn scores loging on')
    injection_group.add_argument('--record-prefix', dest='record_prefix', default=None, type=str,
                                 help='prefix of record filename')
    injection_group.add_argument('--iter', default=1, type=int,
                                 help='Iteration number of FI run.')


    #####
    ##  Quantization Arguments
    ## 
    ##  Quantization is provided by Distiller package;
    ##  Arguments follow the same strategy presented in range_linear.py
    ##  but changing the arguments names to follow torchFI pattern;
    ##  
    #####
    
    str_to_quant_mode_map = {'sym': LinearQuantMode.SYMMETRIC,
                             'asym_s': LinearQuantMode.ASYMMETRIC_SIGNED,
                             'asym_u': LinearQuantMode.ASYMMETRIC_UNSIGNED}

    str_to_clip_mode_map = {'none': ClipMode.NONE, 'avg': ClipMode.AVG, 'n_std': ClipMode.N_STD}

    def from_dict(d, val_str):
        try:
            return d[val_str]
        except KeyError:
            raise argparse.ArgumentTypeError('Must be one of {0} (received {1})'.format(list(d.keys()), val_str))

    linear_quant_mode_str = partial(from_dict, str_to_quant_mode_map)
    clip_mode_str = partial(from_dict, str_to_clip_mode_map)
    
    quantization_group = parser.add_argument_group('Arguments for post-training quantization control')

    quantization_group.add_argument('--quantize', dest='quantize', action='store_true',
                                    help='apply quantization to model')
    quantization_group.add_argument('--quant-mode', dest='quant_mode', type=linear_quant_mode_str, default='sym',
                                    help='linear quantization mode. Choices: ' + ' | '.join(str_to_quant_mode_map.keys()))
    quantization_group.add_argument('--quant-bits-acts', dest='quant_bacts', default=8, type=int, metavar='NUM_BITS',
                                    help='# of bits to quantize features')
    quantization_group.add_argument('--quant-bits-wts', dest='quant_bwts', default=8, type=int, metavar='NUM_BITS',
                                    help='# of bits to quantize weights')
    quantization_group.add_argument('--quant-bits-accum', dest='quant_baccum', default=32, type=int, metavar='NUM_BITS',
                                    help='# of bits of accumulator used during quantization')
    quantization_group.add_argument('--quant-clip-acts', dest='quant_cacts', type=clip_mode_str, default='none',
                                    help='Activations clipping mode. Choices: ' + ' | '.join(str_to_clip_mode_map.keys()))
    quantization_group.add_argument('--quant-clip-n-stds', dest='quant_cnstds', type=float, 
                                    help='When quant-clip-acts is set to \'n_std\', this is the number of standard deviations to use')
    quantization_group.add_argument('--quant-no-clip-layers', dest='quant_noclip_layers', type=str, nargs='+', metavar='LAYER_NAME', default=[], 
                                    help='List of layer names for which not to clip activations. Applicable only if --quant-clip-acts is not \'none\'')
    quantization_group.add_argument('--quant-per-channel', dest='quant_channel', action='store_true', 
                                    help='Enable per-channel quantization of weights (per output channel)')
    quantization_group.add_argument('--quant-scale-approx-bits', dest='quant_scalebits', type=int, metavar='NUM_BITS', 
                                    help='Enables scale factor approximation using integer multiply + bit shift, using this number of bits the integer multiplier')
    quantization_group.add_argument('--quant-stats-file', type=str, metavar='PATH', 
                                    help='Path to YAML file with calibration stats. If not given, dynamic quantization will be run (Note that not all layer types are supported for dynamic quantization)')

    #####
    ##  Pruning Arguments
    #####

    pruning_group = parser.add_argument_group('Arguments for model pruning control')

    pruning_group.add_argument('--pruned', dest='pruned', action='store_true',
                        help='use pruned model')
    pruning_group.add_argument('--prune_compensate', dest='prune_compensate', action='store_true',
                        help='apply an % of faults relative to the amount of weights after prunning')
    pruning_group.add_argument('--pruned_file', metavar='DIR',
                        help='path to pruned checkpoint')
    pruning_group.add_argument('--goldenPred_file', metavar='DIR',
                        help='path to goldenPred_file')


    #####
    ##  GNMT Arguments
    #####

    gnmt_group = parser.add_argument_group('Arguments for model GNMT model')

    gnmt_group.add_argument('--input', help='input file (tokenized)')
    # Replaced by record-prefix
    # TODO: Add fixed prefix folder to save GNMT ouput, record, etc
    # gnmt_group.add_argument('--output', required=True,
    #                         help='output file (tokenized)')
    gnmt_group.add_argument('--model', help='model checkpoint file')
    gnmt_group.add_argument('--reference', default=None,
                            help='full path to the file with reference \
                            translations (for sacrebleu)')

    gnmt_group.add_argument('--beam-size', default=5, type=int,
                            help='beam size')
    gnmt_group.add_argument('--max-seq-len', default=80, type=int,
                            help='maximum prediciton sequence length')
    gnmt_group.add_argument('--cov-penalty-factor', default=0.1, type=float,
                            help='coverage penalty factor')
    gnmt_group.add_argument('--len-norm-const', default=5.0, type=float,
                            help='length normalization constant')
    gnmt_group.add_argument('--len-norm-factor', default=0.6, type=float,
                            help='length normalization factor')

    batch_first_parser = gnmt_group.add_mutually_exclusive_group(required=False)
    
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)


    return parser