"""Test-time refitting of a chunk of one subject set (deviation-statistics stage; inference only).

Same procedure and budget as the generalization runs (vatl_sdf/test_time_fit.py: network frozen, fresh latent and
affine, align 10 epochs, total 36, 50 steps/epoch, horizon 36). Sets come from configs/evaluation/deviation_sets.json;
training subjects (null, calibration) are refitted from scratch exactly like the validation subjects.
Output: fitted_<set>_<chunk>.pt with subject_ids, A, t_mm (effective subject -> template maps) and subject tables.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.generalization.evaluate_generalization import test_frame
from vatl_sdf.test_time_fit import SubjectSampler, build_fitting_model, fit, subject_maps, training_steps_per_epoch

SEED_OFFSET = {'validation': 1, 'null': 3, 'calibration': 4, 'null_extra': 5}  # 1 = the generalization 'test' group


def set_frame(config, frame, sets, name):
    if name == 'validation':
        return test_frame(config, frame.subject_id.tolist(), '', len(sets['validation']))
    ids = sets[name]
    if set(ids) - set(frame.subject_id):
        raise ValueError(f'{name}: IDs outside the checkpoint training subjects')
    return frame.set_index('subject_id', drop=False).loc[ids].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--sets', required=True)
    p.add_argument('--set', required=True, choices=sorted(SEED_OFFSET)); p.add_argument('--output', required=True)
    p.add_argument('--chunk', type=int, default=0); p.add_argument('--chunk-size', type=int, default=10)
    p.add_argument('--align-epochs', type=int, default=10); p.add_argument('--total-epochs', type=int, default=36)
    p.add_argument('--horizon-epochs', type=int); p.add_argument('--steps-per-epoch', type=int)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    started = time.time()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    horizon = a.horizon_epochs or int(checkpoint['scheduler']['T_max'])
    steps = a.steps_per_epoch or training_steps_per_epoch(config)
    sets = json.loads(Path(a.sets).read_text())
    full = set_frame(config, frame, sets, a.set)
    chunk = full.iloc[a.chunk * a.chunk_size:(a.chunk + 1) * a.chunk_size].reset_index(drop=True)
    if not len(chunk):
        raise ValueError('Empty chunk')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    target = out / f'fitted_{a.set}_{a.chunk:02d}.pt'
    if target.exists():
        raise FileExistsError(target)
    model, config = build_fitting_model(checkpoint, len(chunk), a.device)
    sampler = SubjectSampler(chunk, config, seed_offset=SEED_OFFSET[a.set] * 1000 + a.chunk)
    log = (out / f'fit_losses_{a.set}_{a.chunk:02d}.jsonl').open('w')
    n_steps = fit(model, config, sampler, a.align_epochs, a.total_epochs, steps, horizon,
                  log=lambda r: log.write(json.dumps(r) + '\n'))
    log.close()
    A, t = subject_maps(model)
    torch.save({'subject_ids': chunk.subject_id.tolist(), 'set': a.set, 'chunk': a.chunk,
                'settings': {'align_epochs': a.align_epochs, 'total_epochs': a.total_epochs, 'horizon_epochs': horizon,
                             'steps_per_epoch': steps, 'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step']},
                'model_subject_tables': {k: v.detach().cpu() for k, v in model.subject_tensors().items()},
                'A': np.asarray(A), 't_mm': np.asarray(t)}, target)
    print(json.dumps({'set': a.set, 'chunk': a.chunk, 'subjects': len(chunk), 'steps': n_steps, 'seconds': time.time() - started}))


if __name__ == '__main__':
    main()
