#!/usr/bin/env sh


PROJECT=torchfi
export PATH=$PATH:$HOME/${PROJECT}

PROJECT_PATH=$HOME/$PROJECT
DATASET_PATH=$HOME/dataset/imagenet
JOB_PATH=${PROJECT_PATH}/experiments/job.sh

FLAGS="-a resnet50 -e --pretrained --injection --layer=0 -b 256"

EXP_PATH=${PROJECT_PATH}/examples/imagenet
EXP_RUN=resnet_eval.py

# Set job parameters

queue=skl
nodes=1
cpus=112

experiment=${PROJECT}_tst
experiment_out_path=${HOME}/experiment_logs


echo "Creating job ${EXP_RUN} with flags ${FLAGS}"

echo 'Submiting job... '

echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS}"
qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS} 

echo "Job submitted."