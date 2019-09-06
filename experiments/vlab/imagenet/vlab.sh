#!/usr/bin/env sh

PROJECT=torchfi
export PATH=$PATH:$HOME/${PROJECT}

PROJECT_PATH=$HOME/$PROJECT
DATASET_PATH=$HOME/dataset/imagenet
JOB_PATH=${PROJECT_PATH}/experiments/vlab/job.sh


EXP_PATH=${PROJECT_PATH}/examples/imagenet
EXP_RUN=resnet_eval.py

# Get command line args
pruned=$1

# Set job parameters
queue=skl
nodes=1
cpus=8

# Experiments configurations
model=resnet50
niters=5
nlayers=53
bit="random"
location="weights"
pruned_prefix=""

FLAGS_BASE="-a ${model} -e --pretrained --batch-size=1"

if [ ${pruned} -eq 1 ]; then
    pruned_path=$2
    prune_percent=$3
    FLAGS_BASE+=" --pruned --pruned_file=${pruned_path}/resnet50_pruned_${prune_percent}_best.pth.tar"
    prune_compensate=$4
    if [ ${prune_compensate} -eq 1 ]; then
        FLAGS_BASE+=" --prune_compensate"
        pruned_prefix="_pruned_${prune_percent}"
    elif [ ${prune_compensate} -eq 0 ]; then
        pruned_prefix="_pruned_${prune_percent}_nocomp"
    fi
fi

declare -a dtypes=("fp32" "int16" "int8")

## Quantization options
declare -a qbits=(16 8)
declare -a qbitsAccs=(64 32)


for dtype in "${dtypes[@]}"
do
    FLAGS=${FLAGS_BASE}
 
    if [ ${dtype} == "int16" ]; then
        FLAGS+=" --quantize --quant-feats=16 --quant-wts=16 --quant-accum=64"
    elif [ ${dtype} == "int8" ]; then
        FLAGS+=" --quantize --quant-feats=8 --quant-wts=8 --quant-accum=32"
    fi

    outputPath=${HOME}/experiments/iccd
    outputFilePrefix=${outputPath}/${dtype}/${model}
    fidPrefix=${outputFilePrefix}_${dtype}_norm${pruned_prefix}
    experiment_out_path=${outputPath}/${dtype}/logs
    
    experiment_config="golden"
    FLAGS_GOLDEN=${FLAGS}" --golden --prefix-output=${fidPrefix}_${experiment_config}"

    experiment=${model}_${dtype}${pruned_prefix}_${experiment_config}

    echo "Creating job ${experiment} with flags ${FLAGS_GOLDEN}"

    echo 'Submiting job... '

    echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_GOLDEN}"
    qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_GOLDEN}" 

    echo "Job submitted."

    for layer in `seq 0 $nlayers`
    do
        for iter in `seq 1 $niters`
        do
            experiment_config="layer_${layer}_${location}_iter_${iter}"
            FLAGS_FAULTY=${FLAGS}" --faulty --injection --layer=${layer} --${location} --prefix-output=${fidPrefix}_${experiment_config}"

            experiment=${model}_${dtype}${pruned_prefix}_${experiment_config}
            
            echo "Creating job ${experiment} with flags ${FLAGS_FAULTY}"

            echo 'Submiting job... '

            echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_FAULTY}"
            qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_FAULTY}" 

            echo "Job submitted."
        done
    done

done