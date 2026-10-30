#!/bin/bash
# Source this file before manual commands.
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo_work/EUHPC_D32_008/xzhang01/envs/torch/bin/activate
cd /leonardo/home/userexternal/xzhang01/VATL_SDF
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 PYTHONUNBUFFERED=1
