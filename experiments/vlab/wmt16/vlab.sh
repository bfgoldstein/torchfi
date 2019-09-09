#!/usr/bin/env sh

PROJECT=torchfi
export PATH=$PATH:$HOME/${PROJECT}

PROJECT_PATH=$HOME/$PROJECT
DATASET_PATH=$HOME/dataset/wmt16/data/
JOB_PATH=${PROJECT_PATH}/experiments/vlab/wmt16/job.sh
MODEL_PATH=$HOME/models/gnmt

EXP_PATH=${PROJECT_PATH}/examples/wmt16
EXP_RUN=gnmtv2.py

# Set job parameters
queue=skl
nodes=1
cpus=16
mem=8gb

# Experiments configurations
model=gnmtv2
niters=5
nlayers=77
bit="random"
location="weights"

FLAGS_BASE="--batch-size=1" 
FLAGS_BASE+=" --beam-size=10 --cov-penalty-factor=0.1 --len-norm-const=5.0 --len-norm-factor=0.6 --max-seq-len=80 --cov-penalty-factor=0.1 --batch-first"
FLAGS_BASE+=" --input=${DATASET_PATH}newstest2014.tok.clean.bpe.32000.en"
FLAGS_BASE+=" --reference=${DATASET_PATH}newstest2014.de"
FLAGS_BASE+=" --model=${MODEL_PATH}/distiller/model_best.pth"


declare -a dtypes=("fp32")

for dtype in "${dtypes[@]}"
do
    FLAGS=${FLAGS_BASE}

    outputPath=${HOME}/experiments/date
    outputFilePrefix=${outputPath}/${dtype}/${model}
    experiment_out_path=${outputPath}/${dtype}/logs
    
    experiment_config="golden"
    FLAGS_GOLDEN=${FLAGS}" --golden --record-prefix=${outputFilePrefix}_"

    experiment=${model}_${dtype}_${experiment_config}

    echo "Creating job ${experiment} with flags ${FLAGS_GOLDEN}"

    echo 'Submiting job... '

    echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_GOLDEN}"
    qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_GOLDEN}" 

    echo "Job submitted."

    for layer in `seq 0 $nlayers`
    do
        for iter in `seq 1 $niters`
        do
            experiment_config="layer_${layer}_${location}_bit_${bit}_iter_${iter}"
            FLAGS_FAULTY=${FLAGS}" --faulty --injection --layer=${layer} --${location} --iter=${iter} --record-prefix=${outputFilePrefix}_"

            experiment=${model}_${dtype}_${experiment_config}
            
            echo "Creating job ${experiment} with flags ${FLAGS_FAULTY}"

            echo 'Submiting job... '

            echo "qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} ${FLAGS_FAULTY}"
            qsub -V -q ${queue} -lselect=${nodes}:ncpus=${cpus}:mem=${mem} -N ${experiment} -o ${experiment_out_path}/${experiment}.out -e ${experiment_out_path}/${experiment}.err -- ${JOB_PATH} ${EXP_PATH} ${EXP_RUN} ${PROJECT} ${DATASET_PATH} "${FLAGS_FAULTY}" 

            echo "Job submitted."
        done
    done

done