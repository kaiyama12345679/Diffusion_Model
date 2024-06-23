#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1
#$ -o multi-node.out
#$ -e multi-node.err
source /etc/profile.d/modules.sh
module load cuda/12.2/12.2.0
module load hpcx/2.12
cat ${SGE_JOB_HOSTLIST} | awk '{print $0, "slots=4"}' > hostfile
export MASTER_HOST=$(cat ${SGE_JOB_HOSTLIST} | head -n 1) 
export MASTER_IP=$(dig +short $MASTER_HOST)
export EXP_PATH=$(pwd)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run the training script
poetry run deepspeed  \
    --hostfile ./hostfile \
    --launcher OpenMPI \
    --master_addr $MASTER_HOST \
    --master_port 29500 \
    --no_ssh_check \
    ./src/nanodeepspeed.py \
    --deepspeed
echo "All processes are done"