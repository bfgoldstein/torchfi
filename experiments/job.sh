#!/bin/bash

EXP_PATH=$1
EXP_RUN=$2
PROJECT=$3
DATASET_PATH=$4
FLAGS=$5

FLAGS="-a resnet50 -e --pretrained --injection --layer=0 -b 256"

. /export/software/anaconda2/etc/profile.d/conda.sh
conda activate ${PROJECT}

export PATH=$PATH:/opt/pbs/default/bin
export PYTHONPATH=$PYTHONPATH:${HOME}/${PROJECT}

echo ${PYTHONPATH}

echo "cd ${EXP_PATH}"
cd ${EXP_PATH}

echo "python3 ${EXP_RUN} ${FLAGS} ${DATASET_PATH}"

python3 ${EXP_RUN} ${FLAGS} ${DATASET_PATH}