#!/bin/bash

##############################
##
##  Init Variables
##
##############################

debug=0

RED='\033[0;31m'
WARNING='\033[93m'
NC='\033[0m' # No Color

usage()
{
    echo -e "${WARNING}usage: resnet50.sh [[-d path dataset ] [-b batch size] [-g gpu id]] | [-h]] ${NC}"
    echo -e "${WARNING}-d | --dataset ${NC}"
    echo -e "${WARNING}  Path to folder containing imagenet validation set ${NC}"
    echo -e "${WARNING}-b | --batchSize ${NC}"
    echo -e "${WARNING}  Size of the batch during inference ${NC}"
    echo -e "${WARNING}-g | --gpu ${NC}"
    echo -e "${WARNING}  GPU id to run torchfi ${NC}"
    echo -e "${WARNING}-e | --debug ${NC}"
    echo -e "${WARNING}  Activate debug mode. Display run calls wihtout executing it ${NC}"
    echo -e "${WARNING}-h | --help ${NC}"
    echo -e "${WARNING}  Display this help message ${NC}"
}

while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset )        shift
                                dataset=$1
                                ;;
        -b | --batchSize )      shift
                                batchSize=$1
                                ;;
        -g | --gpu )    shift
                                gpu=$1
                                ;;
        -e | --debug )          shift
                                debug=1
                                ;;                                
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ -z "$dataset" -o -z "$batchSize" -o -z "$gpu"]; then
    echo -e "${RED}error: missing arguments${NC}"
    usage
    exit 1
fi

# Add torchfi root folder to LD_LIBRARY_PATH
basedir="$PWD"
export PYTHON_PATH=$basedir:$PYTHON_PATH

# Rercord total excution time
SECONDS=0

# Define which model, # iterations
# and layers for each experiment
model=resnet50
niters=5
nlayers=53
experiment=examples/imagenet/resnet_pruned_eval.py
weightsFile=${basedir}/pytorch_caffe_resenet50_pruned.npy

# Define ouput path to save all results
outputPath=${basedir}/experiments/results/pruned
outputFilePrefix=resnet50_pruned
outputFileExt=.out

echo "Creating ${outputPath}"
mkdir -p ${outputPath}

# log
echo "Running Torchfi experiments on GPU ${gpu} with"
echo "Model: ${model}"
echo "# iterations: ${niters}"
echo "Output path: ${outputPath}"
echo ""

##########################################
####
####  SDC per location (weights or inputs)
####    FP32
####
##########################################

outputFilePrefixBit=${outputPath}/${outputFilePrefix}

fidPrefix=${outputFilePrefixBit}_fp32_golden
echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}"
if [ ${debug} -eq 0 ]; then
    python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --golden --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}
fi
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
                echo "Running SDC per layer iter #${iter}"
                echo "Script: ${experiment}"
                echo "Model: ${model}"
                echo "Layer: ${layer}"
                echo "Bit: ${bit}"
                echo "Location: ${loc}"
                echo "Batch Size: ${batchSize}"
                echo "Datset Path: ${dataset}"
                echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --faulty --injection --layer=${layer} --bit=${bit} --${loc} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}"
                if [ ${debug} -eq 0 ]; then
                    python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --faulty --injection --layer=${layer} --bit=${bit} --${loc} --scores --prefix-output=${fidPrefix} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}
                fi
                echo ""
            done
        done
    done
done


#####################################################
####
####  SDC location
####    quantization (INT16b, INT8b and INT6b)
####
#####################################################

outputFilePrefixLayer=${outputPath}/${outputFilePrefix}

declare -a qbits=(16 8 6)
declare -a qbitsAccs=(64 32 32)

declare -a location=("weights" "features")

for ((i=0; i < ${#qbits[@]}; ++i))
do
    qbit=${qbits[$i]}
    qbitsAcc=${qbitsAccs[$i]}

    fidPrefix=${outputFilePrefixLayer}_int${qbit}_golden
    echo "python3 ${experiment} -a ${model} --evaluate --pretrained --golden --weight_file=${weightsFile} --scores --prefix-output=${fidPrefix} --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}"
    if [ ${debug} -eq 0 ]; then
        python3 ${experiment} -a ${model} --evaluate --pretrained --golden --weight_file=${weightsFile} --scores --prefix-output=${fidPrefix} --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}
    fi
    echo ""

    for layer in `seq 0 $nlayers`
    do
        for bit in `seq 0 $(expr ${qbit} - 1)`
        do
            for loc in "${location[@]}"
            do
                for iter in `seq 1 $niters`
                do
                    fidPrefix=${outputFilePrefixLayer}_int${qbit}_layer_${layer}_bit_${bit}_loc_${loc}_iter_${iter}
                    echo "Running SDC per layer iter #${iter}"
                    echo "Script: ${experiment}"
                    echo "Model: ${model}"
                    echo "Layer: ${layer}"
                    echo "Bit: ${bit}"
                    echo "Location: ${loc}"
                    echo "Quantization: Weights and Acts INT${qbit}b"
                    echo "Quantization: Accumulators INT${qbitsAcc}b"
                    echo "Batch Size: ${batchSize}"
                    echo "Datset Path: ${dataset}"
                    echo "python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --faulty --injection --layer=${layer} --bit=${bit} --${loc} --scores --prefix-output=${fidPrefix} --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}"
                    if [ ${debug} -eq 0 ]; then
                        python3 ${experiment} -a ${model} --evaluate --pretrained --weight_file=${weightsFile} --faulty --injection --layer=${layer} --bit=${bit} --${loc} --scores --prefix-output=${fidPrefix} --quantize --quant-feats=${qbit} --quant-wts=${qbit} --quant-accum=${qbitsAcc} --batch-size=${batchSize} --gpu=${gpu} ${dataset} > ${fidPrefix}${outputFileExt}
                    fi
                    echo ""
                done
            done
        done
    done
done


# output total execution time
duration=$SECONDS
echo "$(($duration / 3600)) hours, $(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."