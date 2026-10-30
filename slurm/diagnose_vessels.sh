#!/bin/bash
# Run inside a GPU allocation after sourcing slurm/environment.sh.
set -euo pipefail
CHECKPOINT=${1:?Usage: bash slurm/diagnose_vessels.sh CHECKPOINT NEW_OUTPUT_DIR SUBJECT_ID}
DESTINATION=${2:?Provide a new output directory}
SUBJECT_ID=${3:?Provide a training subject ID}
if [[ -e "$DESTINATION" ]]; then
  echo "Output already exists: $DESTINATION" >&2
  exit 2
fi
mkdir -p "$DESTINATION"
python -m mra_sdf_only.build_atlas --checkpoint "$CHECKPOINT" \
  --output "$DESTINATION/subject_native" --subject-id "$SUBJECT_ID"
python -m mra_sdf_only.build_atlas --checkpoint "$CHECKPOINT" \
  --output "$DESTINATION/atlas_1mm" --spacing 1 1 1
python -m mra_sdf_only.build_atlas --checkpoint "$CHECKPOINT" \
  --output "$DESTINATION/atlas_05mm" --spacing 0.5 0.5 0.5
