"""Generalization test of a trained checkpoint by test-time fitting (vatl_sdf/test_time_fit.py).

Groups (same procedure, same metrics):
  test        first N validation IDs of the corrected split (YAML order), never seen in training;
  train_refit the 6 Phase-0c training diagnostic subjects, refitted from scratch by the same test-time procedure
              (separates generalization from the fitting budget);
  train       the same 6 subjects decoded with their TRAINED latents/transforms (the 0c training fit).
Metrics: Phase-0c native-grid metrics (Dice, band MAE, volume ratio, component ratio, centreline inside fraction)
at every --eval-epochs (test-time fitting epoch), x1 grid; --final-factors at the last epoch.
Alignment generalization: large-vessel recurrence (1/2/3 mm) of each group's centrelines mapped to the template
by its transforms, against the training subjects mapped by their trained transforms; centroid-translation-only
frame as control.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.alignment.evaluate_alignment import subject_geometry
from analysis.phase0_graph.diagnose_checkpoint import evaluate_subject, select_subjects
from vatl_sdf.alignment_metrics import recurrence_against, summarize_recurrence, transform_statistics
from vatl_sdf.dataset import REQUIRED_COLUMNS
from vatl_sdf.model import SDFModel
from vatl_sdf.splits import read_splits
from vatl_sdf.test_time_fit import SubjectSampler, build_fitting_model, fit, subject_maps, training_steps_per_epoch

METRICS = ('zero_level_dice', 'band_mae_mm', 'volume_ratio_pred_over_ref')
CONNECTIVITY = ('component_ratio_ge_min', 'largest_fraction_difference')


def test_frame(config, train_ids, manifest, count):
    """First `count` validation IDs in YAML order, checked against the manifest and the training set; rows from
    the training TSV with paths resolved exactly as in training."""
    d = config['data']
    splits = read_splits(d['subject_ids'], d['split_key'], d.get('expected_split_counts'))
    ids = splits['val'][:count]
    if manifest:
        listed = json.loads(Path(manifest).read_text())['subject_ids'][:count]
        if listed != ids:
            raise ValueError(f'Manifest selection {listed} differs from YAML val order {ids}')
    if set(ids) & set(train_ids):
        raise ValueError(f'Test subjects in the training set: {sorted(set(ids) & set(train_ids))}')
    if not set(train_ids) <= set(splits['train']):
        raise ValueError('Checkpoint training IDs outside the declared train split')
    frame = pd.read_csv(d['tsv'], sep='\t', dtype={'subject_id': str}).loc[:, REQUIRED_COLUMNS]
    frame = frame.set_index('subject_id', drop=False).loc[ids].reset_index(drop=True)
    parent = Path(d['tsv']).resolve().parent
    for key in ('SDF', 'Graph'):
        frame[key] = frame[key].map(lambda path: str((parent / path).resolve()))
        missing = [p for p in frame[key] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    return frame


def medians(results):
    pick = lambda f: float(np.median([f(r) for r in results]))
    out = {k: pick(lambda r: r['grid_x1'][k]) for k in METRICS}
    out.update({k: pick(lambda r: r['grid_x1']['connectivity'][k]) for k in CONNECTIVITY})
    out['centreline_inside_given_ref_inside'] = pick(lambda r: r['centerline']['pred_inside_given_ref_inside'])
    for label in ('floor', '0.90-1.20', '1.20-1.60', '>=1.60'):
        values = [r['centerline'][label]['pred_inside'] for r in results if r['centerline'][label]['pred_inside'] is not None]
        out[f'centreline_inside_{label}'] = float(np.median(values)) if values else None
    return out


def evaluate_group(model, config, frame, out, factors, save):
    with torch.no_grad():
        return [evaluate_subject(model, config, i, frame.iloc[i], out, factors, save) for i in range(len(frame))]


def run_group(name, checkpoint, frame, a, out):
    """Test-time fit of `frame`; metrics at the evaluation epochs. Returns (summary, maps)."""
    model, config = build_fitting_model(checkpoint, len(frame), a.device)
    sampler = SubjectSampler(frame, config, seed_offset={'test': 1, 'train_refit': 2}[name])
    log_path = out / f'fit_losses_{name}.jsonl'
    log_stream = log_path.open('w')
    per_epoch = {}
    started = time.time()

    def on_epoch_end(epoch, stage):
        if epoch not in a.eval_epochs and epoch != a.total_epochs:
            return
        final = epoch == a.total_epochs
        results = evaluate_group(model, config, frame, out / name / f'epoch_{epoch:03d}',
                                 a.final_factors if final else [1], save=final)
        maps = subject_maps(model)
        per_epoch[epoch] = {'stage': 'align' if stage else 'main', 'elapsed_s': time.time() - started,
                            'median': medians(results), 'transforms': transform_statistics(*maps), 'subjects': results}
        print(json.dumps({'group': name, 'epoch': epoch, **per_epoch[epoch]['median']}), flush=True)
        (out / f'{name}.json').write_text(json.dumps(per_epoch, indent=1))

    steps = fit(model, config, sampler, a.align_epochs, a.total_epochs, a.steps_per_epoch, a.horizon_epochs,
                on_epoch_end=on_epoch_end, log=lambda r: log_stream.write(json.dumps(r) + '\n'))
    log_stream.close()
    maps = subject_maps(model)
    torch.save({'subject_ids': frame.subject_id.tolist(), 'model_subject_tables': {k: v.detach().cpu() for k, v in model.subject_tensors().items()},
                'A': maps[0], 't_mm': maps[1]}, out / f'fitted_{name}.pt')
    return {'steps': steps, 'fit_s': time.time() - started, 'epochs': per_epoch}, maps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--manifest', default='configs/evaluation/validation_cohort.json')
    p.add_argument('--num-test', type=int, default=5)
    p.add_argument('--align-epochs', type=int, default=10); p.add_argument('--total-epochs', type=int, default=36)
    p.add_argument('--horizon-epochs', type=int, help='cosine horizon (default: the checkpoint run length)')
    p.add_argument('--steps-per-epoch', type=int, help='default: per-subject updates per training epoch')
    p.add_argument('--eval-epochs', type=int, nargs='*', default=[10, 12, 15, 20, 28, 36])
    p.add_argument('--final-factors', type=int, nargs='+', default=[1, 2])
    p.add_argument('--recurrence-train-queries', type=int, default=100, help='training subjects used as queries')
    p.add_argument('--skip-train-refit', action='store_true')
    p.add_argument('--device', default='cuda'); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--min-radius-mm', type=float, default=1.2)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    a.horizon_epochs = a.horizon_epochs or int(checkpoint['scheduler']['T_max'])
    a.steps_per_epoch = a.steps_per_epoch or training_steps_per_epoch(config)
    tests = test_frame(config, frame.subject_id.tolist(), a.manifest, a.num_test)
    diag_ids = select_subjects(frame)
    diag = frame.set_index('subject_id', drop=False).loc[diag_ids].reset_index(drop=True)
    settings = {k: v for k, v in vars(a).items()}
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint.get('step'),
               'epoch': checkpoint.get('position', {}).get('epoch'), 'settings': settings,
               'test_subject_ids': tests.subject_id.tolist(), 'train_subject_ids': diag_ids,
               'note': 'screening evidence; test = validation IDs never used in training'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))

    # Trained fit of the diagnostic training subjects (reference row of the train-test gap).
    trained = SDFModel(config, len(frame)).to(a.device); trained.load_state_dict(checkpoint['model']); trained.eval()
    index = [int(frame.index[frame.subject_id == s][0]) for s in diag_ids]
    with torch.no_grad():
        results = [evaluate_subject(trained, config, i, frame.iloc[i], out / 'train_trained', a.final_factors, save=False) for i in index]
    summary['train_trained'] = {'median': medians(results), 'subjects': results}
    train_a, train_t = (x.double().cpu().numpy() for x in trained.maps_mm())
    del trained; torch.cuda.empty_cache() if a.device == 'cuda' else None

    groups = {'test': tests} if a.skip_train_refit else {'test': tests, 'train_refit': diag}
    maps = {}
    for name, rows in groups.items():
        summary[name], maps[name] = run_group(name, checkpoint, rows, a, out)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))

    # Alignment generalization (large-vessel recurrence against the training subjects).
    t0 = time.time()
    with Pool(a.workers) as pool:
        train_geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
        group_geo = {n: pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in rows.iterrows()]) for n, rows in groups.items()}
    learned = lambda geo, A, t: [(lines - centre) @ A[i].T + t[i] for i, (centre, _, lines) in enumerate(geo)]
    centroid = lambda geo: [lines - c for _, c, lines in geo]
    references = {'learned': learned(train_geo, train_a, train_t), 'centroid_translated': centroid(train_geo)}
    q = a.recurrence_train_queries
    recurrence = {f'train_vs_train/{f}': summarize_recurrence(recurrence_against(r[:q], r, exclude=list(range(q))))
                  for f, r in references.items()}
    for name, geo in group_geo.items():
        exclude = None if name == 'test' else index
        clouds = {'learned': learned(geo, *maps[name]), 'centroid_translated': centroid(geo)}
        for f, c in clouds.items():
            fractions = recurrence_against(c, references[f], exclude=exclude)
            recurrence[f'{name}_vs_train/{f}'] = summarize_recurrence(fractions)
        if name == 'test':
            recurrence['test_per_subject/learned_3mm_median'] = {
                s: float(np.median(recurrence_against([cl], references['learned'])[3.0]))
                for s, cl in zip(groups[name].subject_id, clouds['learned'])}
    summary['recurrence'] = recurrence
    summary['recurrence_note'] = (f'queries: 300 large-vessel centreline points per query subject; references: all '
                                  f'{len(frame)} training subjects (trained transforms), the query subject itself excluded; '
                                  f'train_vs_train uses the first {q} training subjects as queries')
    summary['runtime_s'] = {'recurrence': time.time() - t0, 'total': time.time() - started}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'recurrence': {k: v if 'per_subject' in k else {d: round(x['median'], 3) for d, x in v.items()}
                                     for k, v in recurrence.items()}, 'runtime_s': summary['runtime_s']}, indent=1))


if __name__ == '__main__':
    main()
