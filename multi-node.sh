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
cat ${SGE_JOB_HOSTLIST} | awk '{print $0, "slots=4"}' > hostfile
export MASTER_HOST=$(cat ${SGE_JOB_HOSTLIST} | head -n 1) 
export MASTER_IP=$(dig +short $MASTER_HOST)
export EXP_PATH=$(pwd)
export HOST0=$(cat ${SGE_JOB_HOSTLIST} | head -n 1 | tail -n 1)
export HOST1=$(cat ${SGE_JOB_HOSTLIST} | head -n 2 | tail -n 1)
export HOST2=$(cat ${SGE_JOB_HOSTLIST} | head -n 3 | tail -n 1)
export HOST3=$(cat ${SGE_JOB_HOSTLIST} | head -n 4 | tail -n 1)



ssh -p 2222 $HOST0 "module load singularitypro && cd $EXP_PATH && singularity exec --nv --bind /apps:/apps,/usr/share/Modules:/usr/share/Modules diffusion.sif "./command.sh $EXP_PATH $HOST0 0"" &
ssh -p 2222 $HOST1 "module load singularitypro && cd $EXP_PATH && singularity exec --nv --bind /apps:/apps,/usr/share/Modules:/usr/share/Modules diffusion.sif "./command.sh $EXP_PATH $HOST0 1"" &
ssh -p 2222 $HOST2 "module load singularitypro && cd $EXP_PATH && singularity exec --nv --bind /apps:/apps,/usr/share/Modules:/usr/share/Modules diffusion.sif "./command.sh $EXP_PATH $HOST0 2"" &
ssh -p 2222 $HOST3 "module load singularitypro && cd $EXP_PATH && singularity exec --nv --bind /apps:/apps,/usr/share/Modules:/usr/share/Modules diffusion.sif "./command.sh $EXP_PATH $HOST0 3"" &
wait
echo "All processes are done"