#!/bin/bash
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?validation root}
python -m unittest -v tests.test_pipeline 2>&1 | tee "$ROOT/unit_tests_final.log"
python -m tests.check_exports "$ROOT" 2>&1 | tee "$ROOT/export_validation.log"
