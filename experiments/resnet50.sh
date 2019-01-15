#!/bin/bash


experiment=examples/imagenet/resnet_eval.py
model=resnet50
batchSize=256
dataset=/home/bfgoldstein/dataset/imagenet/
niters=1
bits=31
declare -a layers=(0 4 5 6 7 9)

outputPath=/home/bfgoldstein/torchfi_experiments
outputFilePrefixLayer=${outputPath}/layer/resnet50_sdc
outputFilePrefixBit=${outputPath}/bit/resnet50_sdc
outputFileExt=.out

#
#   SDC per Layer 
#
for layer in "${layers[@]}"
do
    for iter in `seq 1 $niters`
    do
        echo "Running SDC per layer iter #${iter}"
        echo "Script: ${experiment}"
        echo "Model: ${model}"
        echo "Layer: ${layer}"
        echo "Bit: rand"
        echo "Batch Size: ${batchSize}"
        echo "Datset Path: ${dataset}"
        echo "python2.7 ${experiment} -a ${model} --evaluate --pretrained --injection --layer=${layer} --batch-size=${batchSize} --gpu=0 ${dataset} &> ${outputFilePrefixLayer}_layer_${layer}_bit_rand_iter_${iter}${outputFileExt}"
        python2.7 ${experiment} -a ${model} --evaluate --pretrained --injection --layer=${layer} --batch-size=${batchSize} --gpu=0 ${dataset} &> ${outputFilePrefixLayer}_layer_${layer}_bit_rand_iter_${iter}${outputFileExt}
    done
done

#
#   SDC per bit position and layer
#
for layer in "${layers[@]}"
do
    for bit in `seq 0 $bits`
    do
        for iter in `seq 1 $niters`
        do
            echo "Running SDC per layer iter #${iter}"
            echo "Script: ${experiment}"
            echo "Model: ${model}"
            echo "Layer: ${layer}"
            echo "Bit: ${bit}"
            echo "Batch Size: ${batchSize}"
            echo "Datset Path: ${dataset}"
            echo "python2.7 ${experiment} -a ${model} --evaluate --pretrained --injection --layer=${layer} --bit=${bit} --batch-size=${batchSize} --gpu=0 ${dataset} &> ${outputFilePrefixBit}_layer_${layer}_bit_${bit}_iter_${iter}${outputFileExt}"
            python2.7 ${experiment} -a ${model} --evaluate --pretrained --injection --layer=${layer} --bit=${bit} --batch-size=${batchSize} --gpu=0 ${dataset} &> ${outputFilePrefixBit}_layer_${layer}_bit_${bit}_iter_${iter}${outputFileExt}
        done
    done
done


#
# Overall SDC
#



#
# SDC per bit position (Overall)
#
