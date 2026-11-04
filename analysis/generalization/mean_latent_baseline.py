"""How much does the individually fitted latent add beyond a latent built only from the subject's age and sex?

Frozen network, no optimization. Unseen validation subjects are decoded on their native grid with their EXISTING
test-time fitted transform and, in place of the fitted latent, a latent derived from the training latents:
  fitted            the existing fitted latent;
  cinema_single     vatl_sdf.inference.weighted_latent (Gaussian age, sigma 2.5 y, closest 100, both sexes), one pass;
  samesex_single    vatl_sdf.conditional_atlas weights (Gaussian age, sigma 5 y, same sex), one pass;
  cinema_age24, samesex_age24   same latents, SDF averaged over 24 weighted age quantiles (conditional_atlas_smooth);
  zero              zero latent (control).
Single-pass variants use the subject's own age and sex condition; age24 variants keep the subject's sex.
The transform was fitted jointly with the fitted latent, so the baselines are not fully independent of the subject,
which favours them relative to a fully subject-independent baseline.
Metrics: evaluate_subject (the generalization metrics), grid x1. Descriptive only: all validation IDs are used.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from analysis.generalization.evaluate_generalization import test_frame
from analysis.phase0_graph.diagnose_checkpoint import evaluate_subject
from vatl_sdf.conditional_atlas import cohort_weights, mean_latent
from vatl_sdf.conditional_atlas_smooth import weighted_age_quantiles
from vatl_sdf.config import condition_vector
from vatl_sdf.deviation import ess
from vatl_sdf.inference import weighted_latent
from vatl_sdf.test_time_fit import build_fitting_model

VARIANTS = ('fitted', 'cinema_single', 'samesex_single', 'cinema_age24', 'samesex_age24', 'zero')
METRICS = ('dice', 'band_mae_mm', 'volume_ratio', 'component_ratio', 'centreline_inside', 'latent_l2')
PRIMARY_SPAN_YEARS, PRIMARY_MAX_SUBJECTS, SECONDARY_SIGMA_YEARS = 5., 100, 5.


def load_fit_tables(paths):
    """Subject IDs and concatenated subject tables of fitted_*.pt files, in the given order."""
    fits = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    ids = [s for f in fits for s in f['subject_ids']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate subjects across fit files')
    names = fits[0]['model_subject_tables']
    return ids, {k: torch.cat([f['model_subject_tables'][k] for f in fits]) for k in names}


def baseline_specs(train_latents, frame, config, age, sex, samples):
    """name -> (latent or None for the fitted one, ages to average or None, weights or None). Inputs are training
    latents and the subject's age and sex only."""
    source = SimpleNamespace(latents=train_latents)
    primary_config = copy.deepcopy(config)
    primary_config['atlas'].update(gaussian_span_years=PRIMARY_SPAN_YEARS, max_subjects=PRIMARY_MAX_SUBJECTS)
    primary_latent, primary = weighted_latent(source, frame, primary_config, age)
    primary = primary.detach().cpu().numpy().astype(np.float64)
    ages, sexes = frame.scan_age.values.astype(float), frame.sex.values
    secondary = cohort_weights(ages, sexes, age, sex, SECONDARY_SIGMA_YEARS)
    secondary_latent = mean_latent(source, secondary)
    return {'fitted': (None, None, None), 'cinema_single': (primary_latent, None, primary),
            'samesex_single': (secondary_latent, None, secondary),
            'cinema_age24': (primary_latent, weighted_age_quantiles(ages, primary, samples), primary),
            'samesex_age24': (secondary_latent, weighted_age_quantiles(ages, secondary, samples), secondary),
            'zero': (torch.zeros_like(train_latents[:1]), None, None)}


def metric_row(result):
    grid = result['grid_x1']
    return {'dice': grid['zero_level_dice'], 'band_mae_mm': grid['band_mae_mm'], 'volume_ratio': grid['volume_ratio_pred_over_ref'],
            'component_ratio': grid['connectivity']['component_ratio_ge_min'],
            'centreline_inside': result['centerline']['pred_inside_given_ref_inside']}


def run(a):
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    train_latents = checkpoint['model']['latents'].float()
    ids, tables = load_fit_tables(a.fits)
    rows = test_frame(config, frame.subject_id.tolist(), '', None)  # all YAML val IDs; raises on overlap with training
    if ids != rows.subject_id.tolist()[:len(ids)]:
        raise ValueError('Fit files are not the leading YAML validation IDs in order')
    chosen = np.array_split(np.arange(len(ids)), a.num_shards)[a.shard]
    model, config = build_fitting_model(checkpoint, len(chosen), a.device)
    config['atlas']['chunk_size'] = a.chunk_size
    with torch.no_grad():
        for name, table in model.subject_tensors().items():
            table.copy_(tables[name][chosen].to(table))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    target = out / f'{a.label}_shard_{a.shard:02d}_of_{a.num_shards:02d}.json'
    if target.exists():
        raise FileExistsError(target)
    records = []
    for local, position in enumerate(chosen):
        row = rows.iloc[int(position)]
        fitted = model.latents.detach()[local].clone()
        specs = baseline_specs(train_latents, frame, config, float(row.scan_age), row.sex, a.age_samples)
        for variant in a.variants:
            latent, ages, weights = specs[variant]
            used = fitted if latent is None else latent[0].to(fitted)
            with torch.no_grad():
                model.latents[local].copy_(used)
                conditions = None if ages is None else [condition_vector(config, x, row.sex) for x in ages]
                result = evaluate_subject(model, config, local, row, out, [1], False, conditions)
                model.latents[local].copy_(fitted)
            records.append({'set': a.label, 'yaml_position': int(position) + 1, 'subject_id': row.subject_id, 'age': float(row.scan_age),
                            'sex': row.sex, 'variant': variant, **metric_row(result), 'latent_l2': float(used.norm()),
                            'ess': None if weights is None else ess(weights),
                            'effective_age': None if weights is None else float(weights @ frame.scan_age.values),
                            'detail': result})
            print(json.dumps({k: v for k, v in records[-1].items() if k != 'detail'}), flush=True)
        target.write_text(json.dumps(records))


def quartiles(values):
    q = np.percentile(values, [25, 50, 75])
    return {'median': float(q[1]), 'q25': float(q[0]), 'q75': float(q[2])}


def paired_differences(table, baseline, metrics=METRICS):
    """Fitted minus baseline per subject, summarized; matched by subject."""
    wide = table.set_index(['subject_id', 'variant'])
    fitted = wide.xs('fitted', level='variant'); other = wide.xs(baseline, level='variant')
    common = fitted.index.intersection(other.index)
    out = {'subjects': len(common)}
    for metric in metrics:
        delta = (fitted.loc[common, metric] - other.loc[common, metric]).astype(float).values
        out[metric] = {**quartiles(delta), 'fitted_higher': int((delta > 0).sum())}
    return out


def summarize(table):
    out = {'subjects': int(table.subject_id.nunique()), 'variants': {}, 'paired_fitted_minus': {}}
    for variant, group in table.groupby('variant'):
        out['variants'][variant] = {m: quartiles(group[m].astype(float).values) for m in METRICS}
    for variant in sorted(set(table.variant) - {'fitted'}):
        out['paired_fitted_minus'][variant] = paired_differences(table, variant)
    return out


def report(a):
    out = Path(a.output)
    records = [r for path in sorted(out.glob('*_shard_*.json')) for r in json.loads(path.read_text())]
    table = pd.DataFrame([{k: v for k, v in r.items() if k != 'detail'} for r in records])
    table.to_csv(out / 'per_subject.tsv', sep='\t', index=False)
    gen, val = table[table.set == 'gen5'], table[table.set == 'val55']
    groups = {'generalization_fit_ids_1_5': gen, 'deviation_fit_ids_1_5': val[val.yaml_position <= 5],
              'ids_1_20': val[val.yaml_position <= 20], 'ids_21_55': val[val.yaml_position > 20], 'ids_1_55': val}
    summary = {name: summarize(group) for name, group in groups.items() if len(group)}
    if a.regression:
        stored = json.loads(Path(a.regression).read_text())['36']['subjects']
        fitted = gen[gen.variant == 'fitted'].set_index('subject_id')
        summary['regression_max_abs_dice_diff'] = max(abs(s['grid_x1']['zero_level_dice'] - fitted.dice[s['subject_id']]) for s in stored)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: {m: round(y[m]['median'], 3) for m in METRICS} for x, y in v['variants'].items()})
                      for k, v in summary.items()}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run'); r.add_argument('--checkpoint', required=True); r.add_argument('--output', required=True)
    r.add_argument('--fits', nargs='+', required=True); r.add_argument('--label', required=True, choices=['gen5', 'val55', 'smoke'])
    r.add_argument('--shard', type=int, default=0); r.add_argument('--num-shards', type=int, default=1)
    r.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    r.add_argument('--age-samples', type=int, default=24); r.add_argument('--chunk-size', type=int, default=400000)
    r.add_argument('--device', default='cuda')
    s = sub.add_parser('report'); s.add_argument('--output', required=True); s.add_argument('--regression')
    a = p.parse_args()
    run(a) if a.command == 'run' else report(a)


if __name__ == '__main__':
    main()
