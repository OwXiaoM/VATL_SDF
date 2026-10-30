# Project handoff

Read `CONTEXT.md` before continuing work. It records the conversation through
2026-10-25, observed results, hypotheses, implemented ablations, and restart steps.
Read `configs/ablations/README.md` for exact experiment commands and caveats.

## Current objective

Investigate fragile/thin/disconnected vessel masks despite smooth MRA atlases.
The user requested controlled experiments they can train one by one, including
SDF-only supervision without MRA intensity supervision. Those configs and scripts
have been implemented and locally checked; full ablation jobs were not submitted.

## Working conventions

- Preserve the original completed run and its outputs.
- Keep ablations controlled: each named variant changes one factor from its
  matched baseline. Start variants from scratch; do not resume them from the
  original baseline checkpoint.
- Distinguish measured evidence from hypotheses. The cause of fragmentation has
  not been established.
- The user intends to launch training one experiment at a time. Do not interpret
  this handoff as a request to submit jobs automatically.
- SDF-only currently means zero intensity loss, not removing the MRA loader,
  coordinate samples, or output head. Explain that distinction when relevant.
- Update `CONTEXT.md` when new jobs, results, conclusions, or decisions arrive.

## Environment

Repository: `/leonardo/home/userexternal/xzhang01/VATL_SDF`.
Use `source slurm/environment.sh` for Python and required HPC modules. Activating
the virtualenv alone did not provide a complete runtime in this session.
This directory did not contain a Git repository when checked.
