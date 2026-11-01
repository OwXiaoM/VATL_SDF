"""Disjoint training-subject sets for the deviation-statistics stage (written once, then frozen in configs/).

From the training IDs of a checkpoint, excluding the Phase-0c diagnostic subjects (kept comparable with earlier
results), a seeded permutation gives: null (LOO null distribution, transforms refitted at test time), calibration
(refitted; recalibration selection), ants_template (reserved for a deferred ANTs template; never queried).
Validation = the first --num-validation YAML validation IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vatl_sdf.splits import read_splits

DIAGNOSTIC = ['IXI269-Guys-0839', 'IXI396-HH-2115', 'IXI521-HH-2353', 'IXI093-HH-1359', 'IXI588-IOP-1158', 'IXI465-HH-2176']


def make_sets(train_ids, validation_ids, sizes, seed, exclude):
    pool = [s for s in train_ids if s not in set(exclude)]
    order = np.random.default_rng(seed).permutation(len(pool))
    out, start = {}, 0
    for name, size in sizes.items():
        out[name] = [pool[i] for i in order[start:start + size]]
        start += size
    if start > len(pool):
        raise ValueError('Sets larger than the available training subjects')
    out['validation'] = list(validation_ids)
    if set(out['validation']) & set(train_ids):
        raise ValueError('Validation IDs overlap the training subjects')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--null', type=int, default=100); p.add_argument('--calibration', type=int, default=50)
    p.add_argument('--ants-template', type=int, default=100); p.add_argument('--num-validation', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    d = config['data']
    splits = read_splits(d['subject_ids'], d['split_key'], d.get('expected_split_counts'))
    train_ids = frame.subject_id.tolist()
    if set(train_ids) != set(splits['train']):
        raise ValueError('Checkpoint training subjects differ from the train split')
    sets = make_sets(train_ids, splits['val'][:a.num_validation],
                     {'null': a.null, 'calibration': a.calibration, 'ants_template': a.ants_template}, a.seed, DIAGNOSTIC)
    record = {'checkpoint': str(Path(a.checkpoint).resolve()), 'seed': a.seed, 'excluded_diagnostic': DIAGNOSTIC,
              'note': 'seeded permutation of training IDs (checkpoint order) minus the diagnostic subjects; '
                      'null, calibration, ants_template are consecutive disjoint blocks; validation = first YAML val IDs',
              **{k: v for k, v in sets.items()}}
    Path(a.output).write_text(json.dumps(record, indent=1))
    print({k: len(v) for k, v in sets.items()})


if __name__ == '__main__':
    main()
