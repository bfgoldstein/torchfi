#!/bin/bash

##########################################################################################
##
##  Script to run experiments on fault injection during training
##
##########################################################################################

debug=$1

if [ ${debug} -eq 1 ]; then
    echo "Debug Mode"
fi

declare -a experiments=("mnist" "cifar10")
project_path=/home/bfgoldstein/torchfi
datasets_path=/home/bfgoldstein/dataset
bits=31
niters=1
gpu=0
workers=6


for experiment_name in "${experiments[@]}"
do
    dataset=${datasets_path}/${experiment_name}/

    if [ ${experiment_name} == "mnist" ]; then
        batchSize=128
        epochs=20
        learningRate=0.01
        momentum=0.5
        gamma=0 # No need of this parameters during mnist training. Setting to zero.
        weightDecay=0 # No need of this parameters during mnist training. Setting to zero.
        testBatchSize=1000
        nlayers=3 # 4 layers
        fiEpoch=10
    elif [ ${experiment_name} == "cifar10" ]; then
        batchSize=128
        epochs=300
        learningRate=0.1
        momentum=0.9
        gamma=0.1
        weightDecay=5e-4
        testBatchSize=100
        nlayers=4 # 5 layers
        fiEpoch=140
    fi

    experiment_path=${project_path}/examples/${experiment_name}/
    experiment=${experiment_name}'_train.py'

    outputPath=${project_path}/experiments/zeus/${experiment_name}/
    prefix=${experiment_name}_fp32_golden
    outputFilePrefix=${outputPath}${prefix}

    echo "python3 ${experiment_path}${experiment} --golden --batch-size=${batchSize} --test-batch-size=${testBatchSize} --epochs=${epochs} --learning-rate=${learningRate} --momentum=${momentum} 
            --gamma=${gamma} --weight-decay=${weightDecay} --plot=${outputFilePrefix}  --record-prefix=${outputFilePrefix}_ --log-path=${outputPath} --log-prefix=${prefix} 
            --workers=${workers} --gpu=${gpu} ${dataset} > ${outputFilePrefix}.out"

    if [ ${debug} -eq 0 ]; then
        python3 ${experiment_path}${experiment} --golden --batch-size=${batchSize} --test-batch-size=${testBatchSize} --epochs=${epochs} --learning-rate=${learningRate} --momentum=${momentum} \
            --gamma=${gamma} --weight-decay=${weightDecay} --plot=${outputFilePrefix}  --record-prefix=${outputFilePrefix}_ --log-path=${outputPath} --log-prefix=${prefix} \
            --workers=${workers} --gpu=${gpu} ${dataset} > ${outputFilePrefix}.out
    fi
    echo ""

    # declare -a location=("weights" "features")
    declare -a location=("weights")

    for layer in `seq 0 $nlayers`
    do
        for bit in `seq 0 $bits`
        do
            for loc in "${location[@]}"
            do
                for iter in `seq 0 $niters`
                do
                    prefix=${experiment_name}_fp32_layer_${layer}_bit_${bit}_epoch_${fiEpoch}_loc_${loc}_iter_${iter}
                    outputFilePrefix=${outputPath}${prefix}
                    echo "Running SDC per layer iter #${iter}"
                    echo "Script: ${experiment}"
                    echo "Model: ${experiment_name}"
                    echo "Layer: ${layer}"
                    echo "Bit: ${bit}"
                    echo "Epoch: ${fiEpoch}"
                    echo "Location: ${loc}"
                    echo "Train Batch Size: ${batchSize}"
                    echo "Test Batch Size: ${testBatchSize}"
                    echo "Datset Path: ${dataset}"
                    echo "python3 ${experiment_path}${experiment} --golden --faulty --injection --layer=${layer} --bit=${bit} --fiEpoch=${fiEpoch} --${loc} 
                            --batch-size=${batchSize} --test-batch-size=${testBatchSize} --epochs=${epochs} --learning-rate=${learningRate} --momentum=${momentum} 
                            --gamma=${gamma} --weight-decay=${weightDecay} --plot=${outputFilePrefix}  --record-prefix=${outputFilePrefix}_ --log-path=${outputPath} --log-prefix=${prefix} 
                            --workers=${workers} --gpu=${gpu} ${dataset} > ${outputFilePrefix}.out"
                        
                    if [ ${debug} -eq 0 ]; then
                        python3 ${experiment_path}${experiment} --golden --faulty --injection --layer=${layer} --bit=${bit} --fiEpoch=${fiEpoch} --${loc} \
                            --batch-size=${batchSize} --test-batch-size=${testBatchSize} --epochs=${epochs} --learning-rate=${learningRate} --momentum=${momentum} \
                            --gamma=${gamma} --weight-decay=${weightDecay} --plot=${outputFilePrefix}  --record-prefix=${outputFilePrefix}_ --log-path=${outputPath} --log-prefix=${prefix} \
                            --workers=${workers} --gpu=${gpu} ${dataset} > ${outputFilePrefix}.out
                    fi
                    echo ""
                done # iter
            done #loc
        done #bit
    done #layer
done # experiment
