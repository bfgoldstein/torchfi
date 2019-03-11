#!/usr/bin/env sh

cores=4
memory=5120
hosts="gpu04 gpu05 gpu06 gpu07 gpu08 gpu09"

pruned=$1

# Define which model, # iterations
# and layers for each experiment
model=resnet50
niters=1
nlayers=53

if [ ${pruned} -eq 0 ]; then
    ##########################################
    ####
    ####  SDC FP32
    ####
    ##########################################

    experiment=examples/imagenet/resnet_eval.py

    # Define ouput path to save all results
    outputPath=${HOME}/experiments/original
    outputFilePrefix=resnet50

    outputFilePrefixBit=${outputPath}/fp32/${outputFilePrefix}

    fidPrefix=${outputFilePrefixBit}_fp32_golden

    args="--golden"
    bsub -q gpu -m "${hosts}" -n ${cores} -W 00:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_fp32_golden -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} ${cores} "${args}" ${fidPrefix}

    echo ""

    bits=31
    declare -a location=("weights" "features")

    for layer in `seq 0 $nlayers`
    do
        for bit in `seq 0 $bits`
        do
            for loc in "${location[@]}"
            do
                for iter in `seq 1 $niters`
                do
                    fidPrefix=${outputFilePrefixBit}_fp32_layer_${layer}_bit_${bit}_loc_${loc}_iter_${iter}
                    echo "Running SDC fp32 #${iter}"
                    echo "Model: ${model}"
                    echo "Layer: ${layer}"
                    echo "Bit: ${bit}"
                    echo "Location: ${loc}"
                    echo "Batch Size: ${batchSize}"

                    args="--faulty --injection --layer=${layer} --bit=${bit} --${loc}"
                    bsub -q gpu -m "${hosts}" -n ${cores} -W 00:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_fp32_faulty_${layer}_bit_${bit}_loc_${loc}_iter_${iter} -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} ${cores} "${args}" ${fidPrefix}

                    echo ""
                done
            done
        done
    done
fi


if [ ${pruned} -eq 1 ]; then
    ##########################################
    ####
    ####  SDC FP32 Pruned
    ####
    ##########################################

    experiment=examples/imagenet/resnet_pruned_eval.py
    weightsFile=${HOME}/torchfi/pytorch_caffe_resenet50_pruned.npy

    # Define ouput path to save all results
    outputPath=${HOME}/experiments/pruned
    outputFilePrefix=resnet50_pruned

    outputFilePrefixBit=${outputPath}/fp32/${outputFilePrefix}

    fidPrefix=${outputFilePrefixBit}_fp32_golden

    args="--weight_file=${weightsFile} --golden"
    bsub -q gpu -m "${hosts}" -n ${cores} -W 00:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_pruned_fp32_golden -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} ${cores} "${args}" ${fidPrefix}

    echo ""

    bits=31
    declare -a location=("weights" "features")

    for layer in `seq 0 $nlayers`
    do
        for bit in `seq 0 $bits`
        do
            for loc in "${location[@]}"
            do
                for iter in `seq 1 $niters`
                do
                    fidPrefix=${outputFilePrefixBit}_fp32_layer_${layer}_bit_${bit}_loc_${loc}_iter_${iter}
                    echo "Running SDC fp32 #${iter}"
                    echo "Model: ${model}"
                    echo "Layer: ${layer}"
                    echo "Bit: ${bit}"
                    echo "Location: ${loc}"
                    echo "Batch Size: ${batchSize}"
                    echo "Weight File: ${weightsFile}"

                    args="--weight_file=${weightsFile} --faulty --injection --layer=${layer} --bit=${bit} --${loc}"
                    bsub -q gpu -m "${hosts}" -n ${cores} -W 00:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_pruned_fp32_faulty_${layer}_bit_${bit}_loc_${loc}_iter_${iter} -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} ${cores} "${args}" ${fidPrefix}

                    echo ""
                done
            done
        done
    done
fi




#####################################################
####
####  SDC quantization 
####    (INT16b and INT8b)
####
#####################################################

# declare -a qbits=(16 8)
# declare -a qbitsAccs=(64 32)

# declare -a location=("weights" "features")

# for ((i=0; i < ${#qbits[@]}; ++i))
# do
#     qbit=${qbits[$i]}
#     qbitsAcc=${qbitsAccs[$i]}
    
#     outputFilePrefixLayer=${outputPath}/int${qbit}/${outputFilePrefix}

#     fidPrefix=${outputFilePrefixLayer}_int${qbit}_golden

#     args="--golden --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc}"
#     bsub -q gpu -m "${hosts}" -n ${cores} -W 01:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_int${qbit}_golden -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} ${cores} "${args}" ${fidPrefix}

#     echo ""

#     for layer in `seq 0 $nlayers`
#     do
#         for bit in `seq 0 $(expr ${qbit} - 1)`
#         do
#             for loc in "${location[@]}"
#             do
#                 for iter in `seq 1 $niters`
#                 do
#                     fidPrefix=${outputFilePrefixLayer}_int${qbit}_layer_${layer}_bit_${bit}_loc_${loc}_iter_${iter}
#                     echo "Running SDC per layer iter #${iter}"
#                     echo "Script: ${experiment}"
#                     echo "Model: ${model}"
#                     echo "Layer: ${layer}"
#                     echo "Bit: ${bit}"
#                     echo "Location: ${loc}"
#                     echo "Quantization: Weights and Acts INT${qbit}b"
#                     echo "Quantization: Accumulators INT${qbitsAcc}b"
#                     echo "Batch Size: ${batchSize}"
#                     echo "Datset Path: ${dataset}"

#                     args="--faulty --injection --layer=${layer} --bit=${bit} --${loc} --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc}"
#                     bsub -q gpu -m "${hosts}" -n ${cores} -W 01:40 -R "rusage[mem=${memory},ngpus_excl_p=1]select[ncc>=3.0]" -R "span[hosts=1]" -J resnet50_int${qbit}_faulty_${layer}_bit_${bit}_loc_${loc}_iter_${iter} -o ${fidPrefix}.out -e ${fidPrefix}.err ./ghpcc_job.sh ${experiment} ${model} "${args}" ${fidPrefix}

#                     echo ""
#                 done
#             done
#         done
#     done
# done
