#!/bin/bash
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?validation root}
python -m unittest -v tests.test_pipeline 2>&1 | tee "$ROOT/unit_tests_final.log"
python -m tests.smoke --device cpu --output "$ROOT/synthetic_midbatch_resume" 2>&1 | tee "$ROOT/synthetic_midbatch_resume.log"
python -m mra_sdf_only.qc_sdf --sdf /leonardo/home/userexternal/xzhang01/data/v1/sdf/IXI002-Guys-0828-Mask.nii.gz --vessel-reference /leonardo/home/userexternal/xzhang01/data/v1/mask/fast/IXI002-Guys-0828-Mask.nii.gz --output "$ROOT/offline_vessel_sign_qc.json"
