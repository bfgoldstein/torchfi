#!/usr/bin/env sh

PROJECT=torchfi
export PATH=$PATH:$HOME/${PROJECT}

PROJECT_PATH=$HOME/$PROJECT
DATASET_PATH=$HOME/dataset/imagenet
JOB_PATH=${PROJECT_PATH}/experiments/vlab/imagenet/job.sh


EXP_PATH=${PROJECT_PATH}/examples/imagenet
EXP_RUN=imagenet_eval.py

# Get command line args
model=$1
nlayers=$2
pruned=$3

# Set job parameters
queue=skl
nodes=1
cpus=16
mem=8gb

# Experiments configurations
# model=resnet50
# nlayers=53
# model=resnet18
# nlayers=20
# model=alexnet
# nlayers=7

niters=5
bit="random"
location="weights"
pruned_prefix=""

FLAGS_BASE="-a ${model} -e --pretrained --batch-size=1"

if [ ${pruned} -eq 1 ]; then
    pruned_path=$4
    prune_percent=$5
    FLAGS_BASE+=" --pruned --pruned_file=${pruned_path}/${model}_pruned_${prune_percent}_best.pth.tar"
    prune_compensate=$6
    if [ ${prune_compensate} -eq 1 ]; then
        FLAGS_BASE+=" --prune_compensate"
        pruned_prefix="_pruned_${prune_percent}"
    elif [ ${prune_compensate} -eq 0 ]; then
        pruned_prefix="_pruned_${prune_percent}_nocomp"
    fi
fi

declare -a dtypes=("fp32" "int16" "int8")


for dtype in "${dtypes[@]}"
do
    FLAGS=${FLAGS_BASE}
 
    if [ ${dtype} == "int16" ]; then
        FLAGS+=" --quantize --quant-mode=sym --quant-bits-acts=16 --quant-bits-wts=16 --quant-bits-accum=64"
    elif [ ${dtype} == "int8" ]; then
        FLAGS+=" --quantize --quant-mode=sym --quant-bits-acts=8 --quant-bits-wts=8 --quant-bits-accum=32"
    fi

    outputPath=${HOME}/experiments/date
    outputFilePrefix=${outputPath}/${dtype}/${model}
    experiment_out_path=${outputPath}/${dtype}/logs
    
    experiment_config="golden"
    FLAGS_GOLDEN=${FLAGS}" --golden --record-prefix=${outputFilePrefix}_"

    experiment=${model}_${dtype}${pruned_prefix}_${experiment_config}

    echo "Creating job ${experiment} with flags ${FLAGS_GOLDEN}"

    echo 'Submiting job... '

    echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_GOLDEN}"
    qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_GOLDEN}" 

    echo "Job submitted."

    for layer in `seq 0 $nlayers`
    do
        for iter in `seq 1 $niters`
        do
            experiment_config="layer_${layer}_${location}_iter_${iter}"
            FLAGS_FAULTY=${FLAGS}" --faulty --injection --layer=${layer} --${location} --iter=${iter} --record-prefix=${outputFilePrefix}_"

            experiment=${model}_${dtype}${pruned_prefix}_${experiment_config}
            
            echo "Creating job ${experiment} with flags ${FLAGS_FAULTY}"

            echo 'Submiting job... '

            echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_FAULTY}"
            qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_FAULTY}" 

            echo "Job submitted."
        done
    done

done