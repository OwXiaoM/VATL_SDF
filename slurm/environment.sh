#!/bin/bash
# Source this file before manual commands.
# ~/.bashrc1 and conda activation are not nounset/errexit-safe (a missing
# command in ~/.bashrc1 returns 127). Relax both while sourcing, then restore
# the caller's options (strict in batch scripts, permissive interactively).
_vatl_shell_options=$(set +o); _vatl_flags=$-  # $(...) drops errexit
set +eu
source ~/.bashrc1
conda activate torch
eval "$_vatl_shell_options"; [[ $_vatl_flags == *e* ]] && set -e
unset _vatl_shell_options _vatl_flags
cd /users/zhangx/VATL_SDF
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 PYTHONUNBUFFERED=1
