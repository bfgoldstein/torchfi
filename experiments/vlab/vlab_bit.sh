#!/usr/bin/env sh

PROJECT=torchfi
export PATH=$PATH:$HOME/${PROJECT}

PROJECT_PATH=$HOME/$PROJECT
DATASET_PATH=$HOME/dataset/imagenet
JOB_PATH=${PROJECT_PATH}/experiments/vlab/job.sh


EXP_PATH=${PROJECT_PATH}/examples/imagenet
EXP_RUN=resnet_eval.py

# Set job parameters
queue=skl
nodes=1
cpus=8

# Experiments configurations
model=resnet50
niters=5
layer=0
nbits=31
location="weights"

FLAGS_BASE="-a ${model} -e --pretrained --batch-size=1"

declare -a dtypes=("fp32" "int16" "int8")

## Quantization options
declare -a qbits=(16 8)
declare -a qbitsAccs=(64 32)


for dtype in "${dtypes[@]}"
do
    FLAGS=${FLAGS_BASE}
 
    if [ ${dtype} == "int16" ]; then
        FLAGS+=" --quantize --quant-feats=16 --quant-wts=16 --quant-accum=64"
        nbits=15
    elif [ ${dtype} == "int8" ]; then
        FLAGS+=" --quantize --quant-feats=8 --quant-wts=8 --quant-accum=32"
        nbits=7
    fi

    outputPath=${HOME}/experiments/iccd/bit
    outputFilePrefix=${outputPath}/${dtype}/${model}
    fidPrefix=${outputFilePrefix}_${dtype}_norm
    experiment_out_path=${outputPath}/${dtype}/logs
    
    experiment_config="golden"
    FLAGS_GOLDEN=${FLAGS}" --golden --prefix-output=${fidPrefix}_${experiment_config}"

    experiment=${model}_${dtype}_${experiment_config}

    echo "Creating job ${experiment} with flags ${FLAGS_GOLDEN}"

    echo 'Submiting job... '

    echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_GOLDEN}"
    qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_GOLDEN}" 

    echo "Job submitted."

    for bit in `seq 0 $nbits`
    do
        for iter in `seq 1 $niters`
        do
            experiment_config="layer_${layer}_bit_${bit}_${location}_iter_${iter}"
            FLAGS_FAULTY=${FLAGS}" --faulty --injection --layer=${layer} --bit=${bit} --${location} --prefix-output=${fidPrefix}_${experiment_config}"

            experiment=${model}_${dtype}_${experiment_config}
            
            echo "Creating job ${experiment} with flags ${FLAGS_FAULTY}"

            echo 'Submiting job... '

            echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_FAULTY}"
            qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_FAULTY}" 

            echo "Job submitted."
        done
    done

done