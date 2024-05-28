#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd
#$ -l USE_SSH=1
#$ -o multi-node.out
#$ -e multi-node.err
source /etc/profile.d/modules.sh
module load singularitypro hpcx-mt/2.12 cuda/12.2/12.2.0 nccl
export PATH=$HOME/.local/bin:$PATH
cat ${SGE_JOB_HOSTLIST} | awk '{print $0, "slots=4"}' > hostfile
export MASTER_HOST=$(cat ${SGE_JOB_HOSTLIST} | head -n 1) 
export MASTER_IP=$(dig +short $MASTER_HOST)
export EXP_PATH=$(pwd)


singularity exec --nv --bind $MODULEPATH:$MODULEPATH diffusion.sif "bash command.sh $MODULEPATH $EXP_PATH $MASTER_IP"