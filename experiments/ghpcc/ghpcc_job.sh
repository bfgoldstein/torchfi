#!/usr/bin/env sh

experiment=$1
model=$2
cores=$3
args=$4
fidPrefix=$5

module load anaconda2/2018.12
source /share/pkg/anaconda2/2018.12/etc/profile.d/conda.sh

conda activate torchfi
unset PYTHONPATH
module load nvidia_cuda_toolkit/9.0/

export PATH=$PATH:/opt/pbs/default/bin
export PYTHONPATH=$PYTHONPATH:${HOME}/torchfi

cd ${HOME}/torchfi

python3 ${experiment} -j ${cores} -a ${model} --evaluate --pretrained ${args} --batch-size=736 --prefix-output=${fidPrefix} --gpu=0 /home/bg74a/datasets/imagenet/