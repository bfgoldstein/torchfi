#!/bin/bash


experiment=examples/imagenet/resnet_pruned_eval.py
model=resnet50
weights=/home/bfgoldstein/resnet50-pruned/pytorch_caffe_resenet50_pruned.npy
batchSize=320
dataset=/home/bfgoldstein/dataset/imagenet/
niters=2
nlayers=53

outputPath=/home/bfgoldstein/torchfi_experiments/pruned
outputFilePrefix=resnet50_pruned
outputFileExt=.out


#
#   SDC per Layer 
#

outputFilePrefixLayer=${outputPath}/layer/${outputFilePrefix}

fidPrefix=${outputFilePrefixLayer}_fp32_golden
echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}"
python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}

for layer in `seq 0 $nlayers`
do
    for iter in `seq 1 $niters`
    do
        fidPrefix=${outputFilePrefixLayer}_fp32_layer_${layer}_bit_rand_loc_rand_iter_${iter}
        echo "Running SDC per layer iter #${iter}"
        echo "Script: ${experiment}"
        echo "Model: ${model}"
        echo "Weights: ${weights}"
        echo "Layer: ${layer}"
        echo "Bit: rand"
        echo "Batch Size: ${batchSize}"
        echo "Datset Path: ${dataset}"
        echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --faulty --injection --layer=${layer} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}"
        python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --faulty --injection --layer=${layer} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}
    done
done

#
#   SDC per bit position and layer
#

outputFilePrefixBit=${outputPath}/bit/${outputFilePrefix}

fidPrefix=${outputFilePrefixBit}_fp32_golden
echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}"
python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}

nlayers=0
bits=31

for layer in `seq 0 $nlayers`
do
    for bit in `seq 0 $bits`
    do
        for iter in `seq 1 $niters`
        do
            fidPrefix=${outputFilePrefixBit}_fp32_layer_${layer}_bit_${bit}_loc_rand_iter_${iter}
            echo "Running SDC per layer iter #${iter}"
            echo "Script: ${experiment}"
            echo "Model: ${model}"
            echo "Weights: ${weights}"
            echo "Layer: ${layer}"
            echo "Bit: ${bit}"
            echo "Batch Size: ${batchSize}"
            echo "Datset Path: ${dataset}"
            echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --faulty --injection --layer=${layer} --bit=${bit} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}"
            python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weights} --faulty --injection --layer=${layer} --bit=${bit} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=0 ${dataset} > ${fidPrefix}${outputFileExt}
        done
    done
done

