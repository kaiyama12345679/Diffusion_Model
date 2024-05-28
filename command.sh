#!/bin/bash
export MODULEPATH=$MODULEPATH:$1
module load hpcx-mt/2.12 cuda/12.2/12.2.0 nccl 
cd $2 && deepspeed --hostfile hostfile --launcher OpenMPI --no_ssh_check --master_addr=$3 ./src/distributed_DDPM.py