#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=ants_time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# ANTs check (CPU): binaries run on a compute node; timing of ONE affine and ONE SyN registration between two
# training subjects' clipped SDFs (+-5 mm; derived images written to OUT only). No template, no cohort.
# Usage: sbatch slurm/ants_timing.sh OUT_DIR FIXED_SDF MOVING_SDF
set -euo pipefail
OUT=${1:?new output dir}; FIXED=${2:?fixed SDF}; MOVING=${3:?moving SDF}
ANTS=/medias/medical/caravel/zhangx/ants-2.6.4
export PATH=$ANTS/bin:$PATH LD_LIBRARY_PATH=$ANTS/lib:${LD_LIBRARY_PATH:-} ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
source /users/zhangx/VATL_SDF/slurm/environment.sh
mkdir -p "$OUT"; cd "$OUT"
antsRegistration --version
python - "$FIXED" "$MOVING" <<'PY'
import sys, nibabel as nib, numpy as np
for name, path in zip(('fixed', 'moving'), sys.argv[1:]):
    image = nib.load(path); sdf = np.asarray(image.dataobj, np.float32)
    sdf = sdf[..., 0] if sdf.ndim == 4 else sdf
    nib.save(nib.Nifti1Image(np.clip(sdf, -5, 5), image.affine), f'{name}_sdf_clip5.nii.gz')
    nib.save(nib.Nifti1Image((sdf < 0).astype(np.uint8), image.affine), f'{name}_mask.nii.gz')
PY
t0=$SECONDS
antsRegistration -d 3 -v 0 -o [affine_,affine_Warped.nii.gz] -n Linear -u 0 \
  -r [fixed_sdf_clip5.nii.gz,moving_sdf_clip5.nii.gz,1] \
  -t Rigid[0.1] -m MeanSquares[fixed_sdf_clip5.nii.gz,moving_sdf_clip5.nii.gz,1,NA] -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox \
  -t Affine[0.1] -m MeanSquares[fixed_sdf_clip5.nii.gz,moving_sdf_clip5.nii.gz,1,NA] -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox
echo "TIMING affine_s $((SECONDS - t0))"
t0=$SECONDS
antsRegistration -d 3 -v 0 -o [syn_,syn_Warped.nii.gz] -n Linear -u 0 -r affine_0GenericAffine.mat \
  -t SyN[0.1,3,0] -m MeanSquares[fixed_sdf_clip5.nii.gz,moving_sdf_clip5.nii.gz,1,NA] -c [100x70x50x20,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox
echo "TIMING syn_s $((SECONDS - t0))"
python - <<'PY'
import nibabel as nib, numpy as np
f = np.asarray(nib.load('fixed_sdf_clip5.nii.gz').dataobj) < 0
for name in ('moving_sdf_clip5.nii.gz', 'affine_Warped.nii.gz', 'syn_Warped.nii.gz'):
    m = np.asarray(nib.load(name).dataobj) < 0
    print('DICE_vs_fixed', name, round(2 * (f & m).sum() / (f.sum() + m.sum()), 3)) if m.shape == f.shape else print('shape differs', name)
PY
echo ANTS_TIMING_OK
