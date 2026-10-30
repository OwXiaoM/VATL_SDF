#!/bin/bash
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?new validation output path}
mkdir -p "$ROOT"
python -m unittest -v tests.test_pipeline 2>&1 | tee "$ROOT/unit_tests.log"
python -m tests.smoke --output "$ROOT/synthetic_cpu" --device cpu 2>&1 | tee "$ROOT/synthetic_cpu.log"
python -m mra_sdf_only.audit_data --limit 3 --output "$ROOT/real_data_audit.jsonl" 2>&1 | tee "$ROOT/real_data_audit.log"
