#!/bin/bash
module load hpcx-mt/2.12 cuda/12.2/12.2.0 nccl
cd $1 && torchrun --rdzv_backend=c10d --rdzv_endpoint=$2 --nproc_per_node=4 --nnodes=4 --node_rank=$3 ./src/distributed_DDPM.py