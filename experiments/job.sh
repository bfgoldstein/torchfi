#!/bin/bash

EXP_PATH=$1
EXP_RUN=$2
PROJECT=$3
FLAGS=$4
DATASET_PATH=$5

export PATH=$PATH:/opt/pbs/default/bin
export PYTHONPATH=$PYTHONPATH:${HOME}${PROJECT}

. /export/software/anaconda2/etc/profile.d/conda.sh
conda activate ${PROJECT}

cd ${EXP_PATH}

python ${EXP_RUN} ${FLAGS} ${DATASET_PATH}