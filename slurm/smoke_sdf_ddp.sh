#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=sdf_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# Phase 1 acceptance: multi-GPU DDP (GPU count from the allocation), full-size model, 8 subjects; 200 steps with periodic
# checkpoints/atlases, exact resume to 300 steps, atlas + native subject export + evaluation.
set -euo pipefail
OUT=${1:?new output directory}
CONFIG=${2:-configs/sdf/00_sdf_baseline.yaml}
shift 2 || shift $#
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
NPROC=${SLURM_GPUS_ON_NODE:-4}
echo "git $(git rev-parse --short HEAD); config $CONFIG; extra: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
COMMON=(data.max_subjects=8 training.checkpoint_every=5 training.atlas_every=5 "output_root=$OUT/runs" "$@")
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" -m vatl_sdf.train_ddp --config "$CONFIG" \
  --set training.max_steps=200 "${COMMON[@]}"
RUN=$(ls -d "$OUT"/runs/*/ | head -1)
CKPT=$(ls "$RUN"/checkpoint_step_00000200.pt)
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" -m vatl_sdf.train_ddp --config "$CONFIG" \
  --set training.max_steps=300 "resume=$CKPT" require_fresh_start=false "${COMMON[@]}"
python -m vatl_sdf.build_atlas --checkpoint "$RUN/checkpoint_step_00000300.pt" --output "$OUT/atlas_1mm" --device cuda
SUBJECT=$(python -c "import torch;print(torch.load('$RUN/checkpoint_step_00000300.pt',map_location='cpu',weights_only=False)['metadata'][0]['subject_id'])")
python -m vatl_sdf.build_atlas --checkpoint "$RUN/checkpoint_step_00000300.pt" --output "$OUT/subject_native" --device cuda --subject-id "$SUBJECT"
SDF=$(python -c "import json;print(json.load(open('$OUT/subject_native/provenance.json'))['SDF'])")
python -m vatl_sdf.evaluate --sdf "$OUT/subject_native/Subject_VesselSDF.nii.gz" --sdf-reference "$SDF" \
  --exported-mask "$OUT/subject_native/Subject_VesselMask.nii.gz" --output "$OUT/subject_native/evaluation.json" | head -40
python - "$RUN" <<'PY'
import json,sys,pathlib
rows=[json.loads(l) for l in (pathlib.Path(sys.argv[1])/'losses.jsonl').read_text().splitlines()]
print('steps logged',len(rows),'first total',rows[0]['total'],'last total',rows[-1]['total'],'last sdf_raw',rows[-1]['sdf_raw'])
assert [r['step'] for r in rows]==list(range(1,len(rows)+1)), 'step sequence broken across resume'
PY
echo SMOKE_OK
