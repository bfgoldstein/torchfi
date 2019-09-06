#!/bin/bash

EXP_PATH=$1
EXP_RUN=$2
PROJECT=$3
DATASET_PATH=$4
FLAGS=$5


. /export/software/anaconda2/etc/profile.d/conda.sh
conda activate ${PROJECT}

export PATH=$PATH:/opt/pbs/default/bin
export PYTHONPATH=$PYTHONPATH:${HOME}/${PROJECT}

echo ${PYTHONPATH}

echo "cd ${HOME}/${PROJECT}"
cd ${HOME}/${PROJECT}

NUM_CORES=8
export MKL_NUM_THREADS=$NUM_CORES
export OMP_NUM_THREADS=$NUM_CORES

echo "MKL_NUM_THREADS = $MKL_NUM_THREADS"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

echo "python3 ${EXP_PATH}/${EXP_RUN} ${FLAGS} ${DATASET_PATH}"

python3 ${EXP_PATH}/${EXP_RUN} ${FLAGS} ${DATASET_PATH}