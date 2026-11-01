"""Scale-gap screening table from post_ablation measurement directories (no computation, reads JSON/TSV only).

Each --runs entry is name=measure_dir, where measure_dir holds diagnostics/, generalization/ and optionally inspect/
(full-scale runs: analysis/20261101_sg_<name>; pilots: pilots/20261101_sg_<name> with post/generalization, post/inspect).
Columns: 0c on the fixed diagnostic subjects (trained latents), test-time fit (test, train_refit) at the final epoch,
large-vessel recurrence at 3 mm (median; references = that run's training subjects), final MoE layer-1 entropy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find(root, name, marker):
    for candidate in (root / name, root / 'post' / name, root):
        if (candidate / marker).exists():
            return candidate
    return None


def row_for(name, root):
    root = Path(root); row = {'run': name}
    diag = find(root, 'diagnostics', 'diagnostics.json')
    if diag:
        d = json.loads((diag / 'diagnostics.json').read_text())
        grids = [s['grid_x1'] for s in d['subjects']]
        row |= {'0c_dice': np.median([g['zero_level_dice'] for g in grids]),
                '0c_mae': np.median([g['band_mae_mm'] for g in grids]),
                '0c_vol': np.median([g['volume_ratio_pred_over_ref'] for g in grids]),
                '0c_comp': np.median([g['connectivity']['component_ratio'] for g in grids]),
                '0c_comp10': np.median([g['connectivity']['component_ratio_ge_min'] for g in grids])}
    gen = find(root, 'generalization', 'summary.json')
    if gen:
        g = json.loads((gen / 'summary.json').read_text())
        for group in ('test', 'train_refit'):
            if group in g:
                final = g[group]['epochs'][str(max(int(e) for e in g[group]['epochs']))]['median']
                row |= {f'{group}_dice': final['zero_level_dice'], f'{group}_mae': final['band_mae_mm'],
                        f'{group}_vol': final['volume_ratio_pred_over_ref'], f'{group}_comp10': final['component_ratio_ge_min']}
        rec = g.get('recurrence', {})
        for key, column in (('train_vs_train/learned', 'rec3_train'), ('test_vs_train/learned', 'rec3_test'),
                            ('test_vs_train/centroid_translated', 'rec3_test_centroid')):
            if key in rec:
                row[column] = rec[key]['3mm']['median']
    inspect = find(root, 'inspect', 'checkpoints.tsv')
    if inspect:
        table = pd.read_csv(inspect / 'checkpoints.tsv', sep='\t')
        if 'moe1_entropy' in table:
            row['moe1_entropy'] = table['moe1_entropy'].iloc[-1]
        if 'latent_norm_median' in table:
            row['latent_norm'] = table['latent_norm_median'].iloc[-1]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='name=measure_dir')
    p.add_argument('--output', help='optional TSV path')
    a = p.parse_args()
    table = pd.DataFrame([row_for(*r.split('=', 1)) for r in a.runs]).set_index('run')
    print(table.round(3).T.to_string())
    if a.output:
        table.to_csv(a.output, sep='\t')


if __name__ == '__main__':
    main()
