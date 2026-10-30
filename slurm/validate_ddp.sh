#!/bin/bash
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?new output directory}
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 -m tests.smoke --real --device cuda --output "$ROOT/real_ddp4" 2>&1 | tee "$ROOT/real_ddp4.log"
