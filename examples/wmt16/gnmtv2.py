#
# This file is part of Distiller project and was developed by:
#  NervanaSystems https://github.com/NervanaSystems/distiller
# 
# Minor changes were applied to satisfy torchFI project needs
# 
# 
# 
# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import print_function

import os
import time
import codecs
import random
import numpy as np
import warnings
from ast import literal_eval
from itertools import zip_longest
import subprocess

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchFI as tfi
from torchFI.injection import FI
import util.parser as tfiParser
from util.record import *
from util.tensor import *

from seq2seq import models
from seq2seq.inference.inference import Translator
from seq2seq.utils import AverageMeter

import seq2seq.data.config as config
from seq2seq.data.dataset import ParallelDataset
from seq2seq.utils import AverageMeter

# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('10.190.0.3', 8097), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


def grouper(iterable, size, fillvalue=None):
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=fillvalue)


def write_output(output_file, lines):
    for line in lines:
        output_file.write(line)
        output_file.write('\n')


def checkpoint_from_distributed(state_dict):
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def main():
    execution_timer = time.time()

    tfiargs = tfiParser.getParser()
    args = tfiargs.parse_args()
    
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True 
        
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        print("Use GPU: {} for training".format(args.gpu))

    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    vocab_size = checkpoint['tokenizer'].vocab_size
    
    model_config = dict(vocab_size=vocab_size, math=checkpoint['config'].math,
                        **literal_eval(checkpoint['config'].model_config))
    
    model_config['batch_first'] = args.batch_first
    
    model = models.GNMT(**model_config)

    state_dict = checkpoint['state_dict']
    
    if checkpoint_from_distributed(state_dict):
        state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

    if args.gpu is not None:
        model = model.cuda()
    
    tokenizer = checkpoint['tokenizer']

    test_data = ParallelDataset(
        src_fname=os.path.join(args.data, config.SRC_TEST_FNAME),
        tgt_fname=os.path.join(args.data, config.TGT_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=0,
        max_len=150,
        sort=False)

    test_loader = test_data.get_loader(batch_size=args.batch_size,
                                       batch_first=True,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=False,
                                       distributed=False)

    translator = Translator(model, tokenizer,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_seq_len,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.gpu is not None)

    model.eval()
    # torch.cuda.empty_cache()
    
    if args.record_prefix is not None:
        record = Record('GNMTv2', batch_size=args.batch_size, injection=args.injection, fiLayer=args.layer, fiFeatures=args.fiFeats, fiWeights=args.fiWeights)
    # Faulty Run
    if args.faulty:
        fi = FI(model, record=record, fiMode=args.injection, fiLayer=args.layer, fiBit=args.bit, 
                fiFeatures=args.fiFeats, fiWeights=args.fiWeights, quantType='SYMMETRIC')
        
        traverse_time = AverageMeter()
        start = time.time()
        fi.traverseModel(model)
        traverse_time.update(time.time() - start)
        
        displayConfig(args)
        fi.injectionMode = True
        print("\n Number of new layers: #%d \n" % fi.numNewLayers)
        
    elif args.golden:
        import distiller.modules as dist
        model = dist.convert_model_to_distiller_lstm(model)
    
    # Setting model to evaluation mode and cuda (if enabled) after FI traverse
    model.eval()
    if args.gpu is not None:
        model = model.cuda()
        
    print(model._modules.items())
    
    test_file = open(args.record_prefix + getRecordPrefix(args, 'fp32', faulty=args.faulty) +
                      ".tok", 'w', encoding='UTF-8')

    batch_time = AverageMeter(False)
    tot_tok_per_sec = AverageMeter(False)
    iterations = AverageMeter(False)
    enc_seq_len = AverageMeter(False)
    dec_seq_len = AverageMeter(False)
    bleu_score = AverageMeter(False)
    score_time = AverageMeter(False)
    stats = {}

    reference_content = readReferenceFile(args)
        
    for batch_idx, (input, target, indices) in enumerate(test_loader):
        translate_timer = time.time()
        input_data, input_lenght = input

        if translator.batch_first:
            batch_size = input_data.size(0)
        else:
            batch_size = input_data.size(1)
        beam_size = args.beam_size

        bos = [translator.insert_target_start] * (batch_size * beam_size)
        bos = torch.LongTensor(bos)
        
        if translator.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        input_lenght = torch.LongTensor(input_lenght)
        stats['total_enc_len'] = int(input_lenght.sum())

        if args.gpu is not None:
            input_data = input_data.cuda(args.gpu, non_blocking=True)
            input_lenght = input_lenght.cuda(args.gpu, non_blocking=True)
            bos = bos.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            context = translator.model.encode(input_data, input_lenght)
            context = [context, input_lenght, None]

            if beam_size == 1:
                generator = translator.generator.greedy_search
            else:
                generator = translator.generator.beam_search
                
            preds, lengths, counter = generator(batch_size, bos, context)
        
        if args.faulty:
            fi.injectionMode = True
        
        stats['total_dec_len'] = lengths.sum().item()
        stats['iters'] = counter

        preds = preds.cpu()
        lengths = lengths.cpu()

        output = []
        for idx, pred in enumerate(preds):
            end = lengths[idx] - 1
            pred = pred[1: end]
            pred = pred.tolist()
            out = translator.tok.detokenize(pred)
            output.append(out)

        output = [output[indices.index(i)] for i in range(len(output))]

        for line_idx, line in enumerate(output):
            score_timer = time.time()
            detok_sentence = detokenizeSentence(args, line)
            chunk = (batch_idx * batch_size) + line_idx
            score = scoreBleuSentence(args, detok_sentence, reference_content[chunk])
            bleu_score.update(score)
            record.addBleuScores(score)
            # Get timing
            elapsed = time.time() - score_timer
            score_time.update(elapsed)
            test_file.write(line)
            test_file.write('\n')

        # Get timing
        elapsed = time.time() - translate_timer
        batch_time.update(elapsed, batch_size)

        total_tokens = stats['total_dec_len'] + stats['total_enc_len']
        ttps = total_tokens / elapsed
        tot_tok_per_sec.update(ttps, batch_size)

        iterations.update(stats['iters'])
        enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
        dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

        if batch_idx % args.print_freq == 0:
            print('[Test {}] Time: {:.3f} ({:.3f})\t   \
                    Decoder iters {:.1f} ({:.1f})\t \
                    Tok/s {:.0f} ({:.0f})\n \
                    Bleu score: {:.2f} ({:.2f})\t \
                    Bleu time: {:.3f} ({:.3f})'.format(
                    batch_idx, batch_time.val, batch_time.avg,
                    iterations.val, iterations.avg,
                    tot_tok_per_sec.val, tot_tok_per_sec.avg,
                    bleu_score.val, bleu_score.avg, 
                    score_time.val, score_time.avg))
            
    # summary timing
    time_per_sentence = (batch_time.avg / batch_size)
    
    print('[Test] Summary \n    \
        Lines translated: {}\t  \
        Avg total tokens/s: {:.0f}\n    \
        Avg time per batch: {:.3f} s\t  \
        Avg time per sentence: {:.3f} ms\n  \
        Avg encoder seq len: {:.2f}\t   \
        Avg decoder seq len: {:.2f}\t   \
        Total decoder iterations: {}\n  \
        Traverse time : {:.3f} s\t  \
        Total number of injections: {}'.format(
        len(test_loader.dataset),
        tot_tok_per_sec.avg,
        batch_time.avg,
        1000 * time_per_sentence,
        enc_seq_len.avg,
        dec_seq_len.avg,
        int(iterations.sum),
        traverse_time.val if args.faulty else 0.0,
        int(fi.numInjections) if args.faulty else 0))

    test_file.close()

    detok = detokenizeFile(args)
    bleu = scoreBleuFile(args, detok)
 
    record.setBleuScoreAvg(bleu)
    saveRecord(args.record_prefix + getRecordPrefix(args, 'fp32', faulty=args.faulty), record)
    
    print('BLEU on test dataset: {}'.format(bleu))
    # Get timing
    execution_elapsed = time.time() - execution_timer
    print('Finished evaluation on test set in {:.2f} seconds'.format(execution_elapsed))


# TODO: Add GNMT arguments
def displayConfig(args):
    # loging configs to screen
    from util.log import logConfig
    logConfig("model", "{}".format("GNMTv2"))
    logConfig("quantization", "{}".format(args.quantize))
    if args.quantize:
        logConfig("mode", "{}".format(args.quant_type))
        logConfig("# bits features", "{}".format(args.quant_bfeats))
        logConfig("# bits weights", "{}".format(args.quant_bwts))
        logConfig("# bits accumulator", "{}".format(args.quant_baccum))
        logConfig("clip", "{}".format(args.quant_clip))
        logConfig("per-channel", "{}".format(args.quant_channel))
    logConfig("injection", "{}".format(args.injection))
    if args.injection:
        logConfig("layer", "{}".format(args.layer))
        logConfig("bit", "{}".format(args.bit))
        logConfig("location:", "  ")
        logConfig("\t features ", "{}".format(args.fiFeats))
        logConfig("\t weights ", "{}".format(args.fiWeights))
        if not(args.fiFeats ^ args.fiWeights): 
            logConfig(" ", "Setting random mode.")
    logConfig("pruned", "{}".format(args.pruned))
    logConfig("prune compensate", "{}".format(args.prune_compensate))
    if args.pruned:
        logConfig("checkpoint from ", "{}".format(args.pruned_file))
    logConfig("batch size", "{}".format(args.batch_size))


def detokenizeFile(args):
    test_path = args.record_prefix + getRecordPrefix(args, 'fp32', faulty=args.faulty)
    # run moses detokenizer
    detok_path = os.path.join(args.data, config.DETOKENIZER)
    detok_test_path = test_path + '.detok'

    with open(detok_test_path, 'w') as detok_test_file, \
            open(test_path + ".tok", 'r') as test_file: 
                subprocess.run(['perl', detok_path], stdin=test_file, 
                               stdout=detok_test_file, stderr=subprocess.DEVNULL)

    return detok_test_path


def detokenizeSentence(args, token_sentence):
    # run moses detokenizer
    detok_path = os.path.join(args.data, config.DETOKENIZERSTRING)
    detoken_sentence = subprocess.run(['perl', detok_path, token_sentence], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    stripped = detoken_sentence.stdout.strip()
    decoded = stripped.decode("utf-8") 
    return decoded


def scoreBleu(detok_file, ref_file):
    # run sacrebleu
    sacrebleu = subprocess.run(['sacrebleu --input {} {} --score-only -lc --tokenize intl'.format(
        detok_file, ref_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if sacrebleu.returncode == 0:
        bleu = float(sacrebleu.stdout.strip())
    else:
        raise Exception(sacrebleu.stderr.decode("utf-8"))
    
    return bleu


def scoreBleuFile(args, detok_test_path):
    reference_path = os.path.join(args.data, config.TGT_TEST_TARGET_FNAME)
    bleu = scoreBleu(detok_test_path, reference_path)
    
    return bleu


def scoreBleuSentence(args, detoken_sentence, reference_sentence):
    import sacreBLEU as sb
    
    bleu = sb.sbleu(detoken_sentence, reference_sentence)
    # fpref = getRecordPrefix(args, 'fp32', faulty=args.faulty) + "_line"
    
    # detoken_file = open(fpref + ".detok", 'w', encoding='UTF-8')
    # detoken_file.write(detoken_sentence)
    # detoken_file.close()
    
    # reference_sentence = reference_sentence.replace('\n', '')
    # reference_file = open(fpref + ".ref", 'w', encoding='UTF-8')
    # reference_file.write(reference_sentence)
    # reference_file.close()
    
    # bleu = scoreBleu(fpref + ".detok", fpref + ".ref")

    return bleu         
              
       
def readReferenceFile(args):
    reference_path = os.path.join(args.data, config.TGT_TEST_TARGET_FNAME)
    with open(reference_path, 'r') as f:
        reference_content = f.readlines()
    
    return reference_content

           
if __name__ == '__main__':
    main()
