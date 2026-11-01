"""Post-hoc recalibration of the probabilistic occupancy atlas (inference only; training objective unchanged).

Motivation (fact, full A4): held-out vessel frequency is ~0.5-0.65 x the atlas occupancy p in 0.01 < p < 0.5
(over-confidence at mid probabilities). Two inference-time remedies, chosen on a CALIBRATION set and reported on
the held-out TEST subjects:
  smoothing      p_s = G_sigma * p (Gaussian, sigma in mm): spreads probability over residual misregistration;
  recalibration  monotone map of p_s fitted on the calibration set: 'platt' (logistic in logit((N p + 1)/(N + 2)))
                 or 'isotonic'.
Calibration set = training subjects whose transforms were REFITTED at test time (fitted_train_refit.pt of the
generalization run), i.e. registered like the test subjects. They are inside the atlas average, which shifts p by at
most 1/N (1/497 = 0.002) per voxel; no leave-one-out correction is applied (stated in the output). The ROI (voxel set
for fitting and evaluation) is fixed across sigma: dilation of (unsmoothed occupancy > 0 or subject vessel) within FOV.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .population_maps import save
from .probabilistic_atlas import calibration, coverage, template_world, subject_labels_in_template


def subject_rows(config, subject_ids):
    rows = pd.read_csv(config['data']['tsv'], sep='\t', dtype={'subject_id': str}).set_index('subject_id', drop=False)
    parent = Path(config['data']['tsv']).resolve().parent
    rows = rows.loc[list(subject_ids)].reset_index(drop=True)
    for key in ('SDF', 'Graph'):
        rows[key] = rows[key].map(lambda path: str((parent / path).resolve()))
    return rows


def labelled_set(fitted, config, occupancy, affine, roi_dilation, min_radius_mm, workers):
    """Per subject: template-space label, FOV mask, fixed ROI and mapped large-vessel centreline points."""
    from analysis.alignment.evaluate_alignment import subject_geometry
    rows = subject_rows(config, fitted['subject_ids'])
    template_mm = template_world(occupancy.shape, affine)
    with Pool(min(workers, len(rows))) as pool:
        geometry = pool.map(subject_geometry, [(row, min_radius_mm) for _, row in rows.iterrows()])
    out = []
    for k, row in enumerate(rows.itertuples()):
        A, t = np.asarray(fitted['A'][k], float), np.asarray(fitted['t_mm'][k], float)
        y, valid = subject_labels_in_template(row.SDF, A, t, template_mm, config['data']['world_bbox_mm'])
        y, valid = y.reshape(occupancy.shape), valid.reshape(occupancy.shape)
        roi = ndi.binary_dilation((occupancy > 0) | y, iterations=roi_dilation) & valid
        centre, _, lines = geometry[k]
        out.append({'subject_id': row.subject_id, 'y': y, 'valid': valid, 'roi': roi, 'lines': (lines - centre) @ A.T + t})
    return out


def logit_feature(p, n):
    s = (n * p + 1.) / (n + 2.)
    return np.log(s) - np.log1p(-s)


class Recalibrator:
    def __init__(self, method, n):
        self.method, self.n = method, n

    def fit(self, p, y):
        if self.method == 'platt':
            self.model = LogisticRegression(C=1e6).fit(logit_feature(p, self.n)[:, None], y)
        elif self.method == 'isotonic':
            self.model = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(p, y)
        return self

    def __call__(self, p):
        if self.method == 'none':
            return p
        shape, flat = p.shape, p.reshape(-1)
        if self.method == 'platt':
            return self.model.predict_proba(logit_feature(flat, self.n)[:, None])[:, 1].reshape(shape)
        return self.model.predict(flat).reshape(shape)

    def describe(self):
        if self.method == 'platt':
            return {'slope': float(self.model.coef_[0, 0]), 'intercept': float(self.model.intercept_[0])}
        return {}


def pooled(subjects, field, key):
    return (np.concatenate([field[s[key]] for s in subjects]), np.concatenate([s['y'][s[key]] for s in subjects]))


def evaluate(subjects, field, n, affine):
    p_roi, y_roi = pooled(subjects, field, 'roi')
    p_fov, y_fov = pooled(subjects, field, 'valid')
    cover = [coverage({'p': field}, s['lines'], affine) for s in subjects]
    return {'roi': calibration(p_roi, y_roi, np.ones(len(p_roi), bool), n),
            'fov': calibration(p_fov, y_fov, np.ones(len(p_fov), bool), n),
            'centreline_coverage_median': {k: float(np.median([c['p'][k] for c in cover if c]))
                                           for k in ('ge_0.1', 'ge_0.25', 'ge_0.5', 'median_p')}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--atlas', required=True, help='probabilistic atlas dir')
    p.add_argument('--calibration-fitted', required=True, help='fitted_train_refit.pt')
    p.add_argument('--test-fitted', required=True, help='fitted_test.pt'); p.add_argument('--output', required=True)
    p.add_argument('--sigmas-mm', type=float, nargs='+', default=[0., 0.5, 1., 1.5, 2., 3.])
    p.add_argument('--methods', nargs='+', default=['none', 'platt', 'isotonic'])
    p.add_argument('--roi-dilation', type=int, default=3); p.add_argument('--min-radius-mm', type=float, default=1.2)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    start = time.time()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    n = len(frame)
    image = nib.load(Path(a.atlas) / 'OccupancyProbability.nii.gz')
    occupancy, affine = np.asarray(image.dataobj, dtype=np.float64), image.affine
    spacing = float(abs(affine[0, 0]))
    cal_fitted = torch.load(a.calibration_fitted, map_location='cpu', weights_only=False)
    test_fitted = torch.load(a.test_fitted, map_location='cpu', weights_only=False)
    if not set(cal_fitted['subject_ids']) <= set(frame.subject_id):
        raise ValueError('Calibration subjects must be training subjects (refitted transforms)')
    if set(test_fitted['subject_ids']) & set(frame.subject_id):
        raise ValueError('Test subjects overlap the training subjects')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    cal = labelled_set(cal_fitted, config, occupancy, affine, a.roi_dilation, a.min_radius_mm, a.workers)
    test = labelled_set(test_fitted, config, occupancy, affine, a.roi_dilation, a.min_radius_mm, a.workers)
    results, best = [], None
    for sigma in a.sigmas_mm:
        smoothed = ndi.gaussian_filter(occupancy, sigma / spacing) if sigma > 0 else occupancy
        p_cal, y_cal = pooled(cal, smoothed, 'roi')
        for method in a.methods:
            recal = Recalibrator(method, n).fit(p_cal, y_cal) if method != 'none' else Recalibrator('none', n)
            field = recal(smoothed)
            row = {'sigma_mm': sigma, 'method': method, 'parameters': recal.describe(),
                   'calibration_set': evaluate(cal, field, n, affine), 'test': evaluate(test, field, n, affine)}
            results.append(row)
            score = row['calibration_set']['roi']['loglik_per_voxel']
            print(json.dumps({'sigma_mm': sigma, 'method': method, 'cal_loglik': score,
                              'test_loglik': row['test']['roi']['loglik_per_voxel'],
                              'test_brier_skill': row['test']['roi']['brier_skill'],
                              'test_ece': row['test']['roi']['expected_calibration_error']}), flush=True)
            if best is None or score > best[0]:
                best = (score, sigma, method, field)
    _, sigma, method, field = best
    save(field, affine, out / f'OccupancyCalibrated_sigma{sigma:g}mm_{method}.nii.gz')
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'], 'subjects': n,
               'atlas': str(Path(a.atlas).resolve()), 'calibration_subjects': cal_fitted['subject_ids'],
               'test_subjects': test_fitted['subject_ids'],
               'note': f'calibration subjects are inside the atlas average (bias <= 1/N = {1 / n:.4f}); selection by '
                       'calibration-set ROI log-likelihood; test numbers are reported, not used for selection',
               'selected': {'sigma_mm': sigma, 'method': method}, 'results': results, 'runtime_s': time.time() - start}
    (out / 'atlas_calibration.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
