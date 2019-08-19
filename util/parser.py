import argparse
import torchvision.models as models


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
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
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


    #####
    ##  Quantization Arguments
    #####
    
    quantization_group = parser.add_argument_group('Arguments for post-training quantization control')

    quantization_group.add_argument('--quantize', dest='quantize', action='store_true',
                                    help='apply quantization to model')
    quantization_group.add_argument('--quant-type', dest='quant_type', default='SYMMETRIC', type=str,
                                    help='Type of quantization: "sym", "asym_u", "asym_s"')
    quantization_group.add_argument('--quant-feats', dest='quant_bfeats', default=8, type=int,
                                    help='# of bits to quantize features')
    quantization_group.add_argument('--quant-wts', dest='quant_bwts', default=8, type=int,
                                    help='# of bits to quantize weights')
    quantization_group.add_argument('--quant-accum', dest='quant_baccum', default=32, type=int,
                                    help='# of bits of accumulator used during quantization')
    quantization_group.add_argument('--quant-clip', dest='quant_clip', action='store_true',
                                    help='enable clipping of features during quantization')
    quantization_group.add_argument('--quant-channel', dest='quant_channel', action='store_true',
                                    help='enable per-channel quantization of weights')


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

    return parser