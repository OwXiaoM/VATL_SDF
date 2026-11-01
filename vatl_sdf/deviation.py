"""Deviation of individual subjects from the population, with a leave-one-out null (inference only).

Alignment methods (one factor: subject -> template alignment); every population statistic of a method is built from
the training subjects' REAL labels (SDF < 0) and graphs mapped by that method:
  learned   training subjects by their trained transforms; queries by test-time REFITTED transforms;
  centroid  A = I, vessel centroid moved onto the mean training centroid (learned template frame);
  decoder   separate row: occupancy = mean of the training subjects' template-space decodings (per-subject maps saved
            by `probabilistic_atlas shard --save-subjects`); queries mapped by the learned (refitted) transforms.
Queries: null set (training subjects, LOO: own contribution removed exactly), calibration set (training subjects,
LOO; used only to select the recalibration) and validation subjects (never used for any selection or fit).
Population weights: uniform, or a Gaussian kernel on age (sigma in years); the query itself always has weight 0.
Smoothed probabilities use one pseudo-subject: (sum w y + c) / (sum w + 2c), c = sum w / ESS (uniform: (k+1)/(n+2)).

Per query, method and weighting:
  voxels      ROI/FOV Bernoulli log-likelihood and Brier of the query label under the occupancy map (uncalibrated;
              calibrated = Gaussian smoothing + monotone map selected on the calibration set), centreline coverage
              (large-vessel centreline, occupancy >= 0.1 / 0.25 / 0.5);
  edges       per population edge presence (centreline within r of >= edge_cover of the edge) vs p_e, surprise
              -log p(observed) and total;
  connectivity per edge with both nodes present, surprise of connected / not under graph and mask criteria, totals;
  radius      per node -SDF (native, x det(A)^(1/3)) and graph radius: abs error, robust z (IQR/1.349), percentile;
              nodes whose graph radius is at the floor (<= 0.801 mm) reported separately;
  affine      (learned) squared Mahalanobis distance of [A | t] to the training transforms.
Empirical p = (1 + #null >= s) / (1 + #null) (null queries: other null queries only); per-element p-values with
Benjamini-Hochberg within subject and family; subject-level false-positive rates at alpha with Holm over the
primary score family.
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
from scipy.spatial import cKDTree

from .atlas_calibration import Recalibrator
from .graph_atlas import build_population, held_out_rows, presence
from .graph_coords import edge_samples, load_graph, world_to_voxel
from .inference import world_grid
from .model import SDFModel
from .population_maps import save
from .probabilistic_atlas import calibration, subject_labels_in_template, template_world, within_radius
from .vessel_attribute_atlas import FLOOR_MM, subject_attributes

THRESHOLDS = (0.1, 0.25, 0.5)
STATE = {}  # read-only data shared with forked workers


# ---------------------------------------------------------------- weights and weighted statistics
def kernel_weights(ages, age, sigma, exclude=None):
    w = np.ones(len(ages)) if sigma is None else np.exp(-0.5 * ((np.asarray(ages, float) - age) / sigma) ** 2)
    if exclude is not None:
        w[exclude] = 0.
    return w


def ess(w):
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def smoothed(weighted_k, weighted_n, w):
    c = w.sum() / ess(w)
    return (weighted_k + c) / (weighted_n + 2 * c)


def bernoulli_surprise(p, observed):
    return np.where(observed, -np.log(p), -np.log1p(-p))


def weighted_quantiles(values, w, qs):
    order = np.argsort(values); v, w = values[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cdf, v)


def radius_deviation(values, distribution, w, min_subjects=3):
    """values (k,), distribution (subjects, k) NaN = absent: abs error, robust z, weighted percentile and a
    population-referenced two-sided p-value min(1, 2 min(P(X <= v), P(X >= v))) (one pseudo-subject) per node."""
    out = np.full((len(values), 4), np.nan)
    for k, v in enumerate(values):
        ok = np.isfinite(distribution[:, k]) & (w > 0)
        if not np.isfinite(v) or ok.sum() < min_subjects:
            continue
        col, wk = distribution[ok, k], w[ok]
        q1, med, q3 = weighted_quantiles(col, wk, [0.25, 0.5, 0.75])
        c = wk.sum() / ess(wk)
        tail = min(wk[col <= v].sum() + c, wk[col >= v].sum() + c) / (wk.sum() + c)
        out[k] = abs(v - med), (v - med) / max((q3 - q1) / 1.349, 1e-6), wk[col <= v].sum() / wk.sum(), min(1., 2 * tail)
    return out


def outcome_p(p_success, observed):
    """Population-referenced p-value of a binary outcome: its probability if it is the less likely one, else 1."""
    q = np.where(observed, p_success, 1 - p_success)
    return np.where(q < 0.5, q, 1.)


def affine_vector(A, t):
    return np.r_[np.asarray(A, float).ravel(), np.asarray(t, float)]


def mahalanobis2(x, X, w):
    mu = (w[:, None] * X).sum(0) / w.sum()
    d = X - mu
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / w.sum()
    cov += 1e-6 * np.trace(cov) / len(x) * np.eye(len(x))
    e = x - mu
    return float(e @ np.linalg.solve(cov, e))


# ---------------------------------------------------------------- occupancy stacks
class Stack:
    """Per-subject binary maps on the population support (flat indices); optional per-subject validity."""

    def __init__(self, labels, valid, support):
        self.support = support
        self.Y = np.stack([np.unpackbits(x, count=support.max() + 1)[support] for x in labels]).astype(np.float32) if len(labels) else None
        self.V = np.stack([np.unpackbits(x, count=support.max() + 1)[support] for x in valid]).astype(np.float32) if valid is not None else None
        if self.V is not None:  # labels count only inside the subject's field of view
            self.Y *= self.V
        self.sum_y = self.Y.sum(0, dtype=np.float64)
        self.sum_v = self.V.sum(0, dtype=np.float64) if self.V is not None else None

    def values(self, w, uniform_exclude=None):
        """Weighted occupancy on the support; uniform weights with one excluded subject use precomputed sums."""
        if uniform_exclude is not None:
            num = self.sum_y - (self.Y[uniform_exclude] if uniform_exclude >= 0 else 0)
            den = (self.sum_v - (self.V[uniform_exclude] if uniform_exclude >= 0 else 0)) if self.V is not None else \
                float(len(self.Y) - (uniform_exclude >= 0))
        else:
            w32 = w.astype(np.float32)
            num = (w32 @ self.Y).astype(np.float64)
            den = (w32 @ self.V).astype(np.float64) if self.V is not None else float(w.sum())
        return np.where(np.asarray(den) > 0, num / np.maximum(den, 1e-12), 0.)

    def full(self, values, n_voxels):
        out = np.zeros(n_voxels)
        out[self.support] = values
        return out


# ---------------------------------------------------------------- per-subject preparation (workers)
def subject_preparation(args):
    """Template-space label/validity (bit-packed) under each method; large-vessel and all-radius centrelines."""
    from analysis.alignment.evaluate_alignment import subject_geometry
    row, transforms, min_radius = args
    centre, centroid, lines = subject_geometry((row, min_radius))
    graph = load_graph(row['Graph'])
    all_points = edge_samples(graph['pos'], graph['edges'], 0.4)[0] if len(graph['edges']) else np.zeros((0, 3))
    template_mm, shape, bbox = STATE['template_mm'], STATE['shape'], STATE['bbox']
    out = {'subject_id': row['subject_id'], 'centre': centre, 'centroid': centroid, 'methods': {}}
    for method, (A, t) in transforms.items():
        if t is None:  # centroid: computed here from the subject centroid
            t = STATE['mean_centroid'] - (centroid - centre)
        y, valid = subject_labels_in_template(row['SDF'], A, t, template_mm, bbox)
        out['methods'][method] = {'A': A, 't': t, 'y': np.packbits(y), 'valid': np.packbits(valid),
                                  'lines': (lines - centre) @ A.T + t, 'all_lines': (all_points - centre) @ A.T + t}
    return out


def attributes_worker(args):
    row, A, t, method = args
    return subject_attributes((row, A, t, STATE['populations'][method], STATE['params']))


# ---------------------------------------------------------------- scoring of one query
def voxel_scores(p, y, valid, n_eff, recal=None):
    roi = ndi.binary_dilation((STATE['support_mask']) | y, iterations=STATE['roi_dilation']) & valid
    out = {}
    for name, field in (('uncal', p),) + ((('cal', recal(p)),) if recal is not None else ()):
        r = calibration(field, y, roi, n_eff); f = calibration(field, y, valid, n_eff)
        out[f'nll_roi_{name}'] = -r['loglik_per_voxel']; out[f'brier_roi_{name}'] = r['brier']
        out[f'nll_fov_{name}'] = -f['loglik_per_voxel']
    return out


def coverage_scores(fields, points):
    out = {}
    if not len(points):
        return out
    vox = world_to_voxel(STATE['affine'], points).T
    for name, field in fields.items():
        values = ndi.map_coordinates(field.reshape(STATE['shape']), vox, order=1, mode='constant', cval=0.)
        out.update({f'uncov_{t:g}_{name}': float((values < t).mean()) for t in THRESHOLDS})
    return out


def score_query(task):
    """task: (query key, method, weighting name, sigma, save_maps). Returns subject scores and element arrays."""
    key, method, wname, sigma, save_maps = task
    S = STATE
    q = S['queries'][key]
    exclude = S['train_index'].get(q['subject_id'])
    w = kernel_weights(S['ages'], q['age'], sigma, exclude)
    n_eff = ess(w)
    frame_method = 'learned' if method == 'decoder' else method
    m = q['methods'][frame_method]
    n_vox = int(np.prod(S['shape']))
    valid = np.unpackbits(m['valid'], count=n_vox).astype(bool)
    y = np.unpackbits(m['y'], count=n_vox).astype(bool) & valid
    stack = S['stacks'][method]
    values = stack.values(w, uniform_exclude=(exclude if exclude is not None else -1) if sigma is None else None)
    p = stack.full(values, n_vox)
    cal = S['calibrators'].get((method, wname))
    recal = None
    if cal is not None:
        sig, rc = cal
        recal = (lambda f: rc(ndi.gaussian_filter(f.reshape(S['shape']), sig).ravel() if sig > 0 else f))
    scores = {'ess': n_eff, **voxel_scores(p, y, valid, n_eff, recal)}
    scores.update(coverage_scores({'uncal': p} | ({'cal': recal(p)} if recal else {}), m['lines']))
    elements = {}
    if save_maps:
        s = np.where(y, -np.log((n_eff * p + 1) / (n_eff + 2)), -np.log1p(-(n_eff * p + 1) / (n_eff + 2))) * valid
        elements['voxel_surprise_map'] = s.astype(np.float32)
    if method != 'decoder':
        attr = S['attributes'][method][key]
        tr = S['train_attributes'][method]
        if method == 'learned':
            X = S['train_affine']
            keep = w > 0
            scores['affine_d2'] = mahalanobis2(affine_vector(m['A'], m['t']), X[keep], w[keep])
        for pop_key, pop in S['populations'][method].items():
            tag = pop_key.split('_')[0]  # r3mm / r2mm
            Ytr = S['presence'][method][pop_key]
            trees = cKDTree(m['lines']) if len(m['lines']) else None
            yq = presence([trees], pop['edge_points_mm'], pop['radius_mm'], pop['edge_cover'])[0]
            pe = smoothed(w @ Ytr, w.sum(), w)
            surprise = bernoulli_surprise(pe, yq)
            scores[f'edge_total_{tag}'] = float(surprise.sum())
            obs = attr['populations'][pop_key]
            both_tr = np.array([r['populations'][pop_key]['both'] for r in tr], float)
            conn, conn_p = {}, {}
            for crit in ('graph', 'mask'):
                g_tr = np.array([r['populations'][pop_key][crit] for r in tr], float)
                pc = smoothed(w @ (both_tr * g_tr), w @ both_tr, w)
                sc = np.where(obs['both'], bernoulli_surprise(pc, obs[crit]), np.nan)
                scores[f'conn_{crit}_total_{tag}'] = float(np.nansum(sc))
                conn[crit] = sc
                conn_p[crit] = np.where(obs['both'], outcome_p(pc, obs[crit]), np.nan)
            node_tr = np.array([r['populations'][pop_key]['node_radius'] for r in tr])  # (subjects, nodes, 2)
            scale_tr = np.array([r['scale'] for r in tr])[:, None]
            floor = obs['node_radius'][:, 1] <= FLOOR_MM
            dev = {'sdf': radius_deviation(obs['node_radius'][:, 0], node_tr[:, :, 0], w),
                   'sdf_scaled': radius_deviation(obs['node_radius'][:, 0] * attr['scale'], node_tr[:, :, 0] * scale_tr, w),
                   'graph': radius_deviation(obs['node_radius'][:, 1], node_tr[:, :, 1], w)}
            for name, d in dev.items():
                for part, sel in (('', ~floor), ('_floor', floor)):
                    z = np.abs(d[sel, 1]); z = z[np.isfinite(z)]
                    scores[f'radius_{name}{part}_absz_{tag}'] = float(np.median(z)) if len(z) else np.nan
                    e = d[sel, 0]; e = e[np.isfinite(e)]
                    scores[f'radius_{name}{part}_abserr_{tag}'] = float(np.median(e)) if len(e) else np.nan
            elements[pop_key] = {'edge_surprise': surprise, 'edge_present': yq, 'edge_p': pe,
                                 'conn_graph': conn['graph'], 'conn_mask': conn['mask'],
                                 'radius_absz': np.where(floor, np.nan, np.abs(dev['sdf'][:, 1])),
                                 'radius_z': dev['sdf'][:, 1], 'radius_percentile': dev['sdf'][:, 2], 'floor': floor,
                                 'pop_p_edge': outcome_p(pe, yq), 'pop_p_conn_graph': conn_p['graph'],
                                 'pop_p_conn_mask': conn_p['mask'], 'pop_p_radius': np.where(floor, np.nan, dev['sdf'][:, 3])}
            if pop_key == S['primary_population']:
                elements[pop_key]['defects'] = classify_absent(pop, yq, pe, m, y)
    return key, method, wname, scores, elements


def classify_absent(pop, yq, pe, m, y):
    """Population edges expected (p_e >= 0.5) but absent in the subject: which nearby evidence exists, in order
    radius_threshold (all-radius graph centreline within r) > graph_defect (mask vessel within r, no graph) >
    displaced (large-vessel centreline within 2r) > absent (none of these)."""
    r, cover = pop['radius_mm'], pop['edge_cover']
    labels = []
    mask_near = ndi.binary_dilation(y.reshape(STATE['shape']), STATE['ball'](r))
    for k in np.flatnonzero(~yq & (pe >= 0.5)):
        pts = pop['edge_points_mm'][k]
        frac = lambda cloud, rad: (cKDTree(cloud).query(pts, distance_upper_bound=rad)[0] <= rad).mean() if len(cloud) else 0.
        vox = np.rint(world_to_voxel(STATE['affine'], pts)).astype(int)
        inside = (vox >= 0).all(1) & (vox < STATE['shape']).all(1)
        mask_frac = mask_near[tuple(vox[inside].T)].sum() / len(pts)
        if frac(m['all_lines'], r) >= cover:
            labels.append('radius_threshold')
        elif mask_frac >= cover:
            labels.append('graph_defect')
        elif frac(m['lines'], 2 * r) >= cover:
            labels.append('displaced')
        else:
            labels.append('absent')
    return labels


# ---------------------------------------------------------------- empirical p-values, FDR, FPR
def empirical_p(value, null_values):
    null_values = np.asarray(null_values, float); null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(value) or not len(null_values):
        return np.nan
    return (1. + (null_values >= value).sum()) / (1. + len(null_values))


def benjamini_hochberg(p, q):
    p = np.asarray(p, float); ok = np.isfinite(p)
    reject = np.zeros(len(p), bool)
    if ok.any():
        pv = p[ok]; order = np.argsort(pv); m = len(pv)
        passed = np.flatnonzero(pv[order] <= q * np.arange(1, m + 1) / m)
        if len(passed):
            idx = np.flatnonzero(ok)[order[:passed.max() + 1]]
            reject[idx] = True
    return reject


def holm(p, alpha):
    p = np.asarray(p, float); ok = np.isfinite(p)
    reject = np.zeros(len(p), bool)
    order = np.flatnonzero(ok)[np.argsort(p[ok])]
    for rank, i in enumerate(order):
        if p[i] > alpha / (len(order) - rank):
            break
        reject[i] = True
    return reject


def primary_scores(method, available):
    nll = 'nll_roi_cal' if 'nll_roi_cal' in available else 'nll_roi_uncal'
    if method == 'decoder':
        return [nll, 'uncov_0.25_uncal']
    out = [nll, 'uncov_0.25_uncal', 'edge_total_r3mm', 'conn_graph_total_r3mm', 'conn_mask_total_r3mm', 'radius_sdf_absz_r3mm']
    return out + (['affine_d2'] if method == 'learned' else [])


ELEMENT_FAMILIES = ('edge_surprise', 'conn_graph', 'conn_mask', 'radius_absz')
POPULATION_P = {'edge_surprise': 'pop_p_edge', 'conn_graph': 'pop_p_conn_graph', 'conn_mask': 'pop_p_conn_mask',
                'radius_absz': 'pop_p_radius'}


def null_pvalues(results, method, wname, keys_null, keys_eval, alpha, q):
    """Subject and element p-values of evaluation keys against the null keys (null keys: leave-self-out).
    Subject level: per-score empirical p; Holm over the primary family; family min-p calibrated on the null
    (p_family = (1 + #null with min-p <= own) / (1 + #null); full resolution unlike Holm with ~100 null subjects).
    Element level, BH within family: (a) empirical p against the null subjects' same element (resolution 1/#null);
    (b) population-referenced p (outcome probability / two-sided radius tail, resolution ~1/ESS)."""
    scores = {k: results[(k, method, wname)][0] for k in keys_null + keys_eval}
    names = sorted({s for k in scores for s, v in scores[k].items() if isinstance(v, float)} - {'ess'})
    out = {}
    for k in dict.fromkeys(keys_null + keys_eval):
        null = [j for j in keys_null if j != k]
        pv = {s: empirical_p(scores[k].get(s, np.nan), [scores[j].get(s, np.nan) for j in null]) for s in names}
        prim = primary_scores(method, scores[k])
        rec = {'p': pv, 'holm_any': bool(holm([pv[s] for s in prim], alpha).any()), 'primary': prim, 'elements': {}}
        el = results[(k, method, wname)][1]
        for pop_key in [x for x in el if x.startswith('r')]:
            fam = {}
            for f in ELEMENT_FAMILIES:
                v = el[pop_key][f]
                nulls = np.array([results[(j, method, wname)][1][pop_key][f] for j in null])
                p_el = np.array([empirical_p(v[i], nulls[:, i]) for i in range(len(v))])
                p_pop = np.asarray(el[pop_key][POPULATION_P[f]], float)
                fam[f] = {'p': p_el, 'reject': benjamini_hochberg(p_el, q),
                          'p_population': p_pop, 'reject_population': benjamini_hochberg(p_pop, q)}
            rec['elements'][pop_key] = fam
        rec['min_p'] = float(np.nanmin([pv[s] for s in prim]))
        out[k] = rec
    for k, rec in out.items():
        null = [j for j in keys_null if j != k]
        rec['p_family'] = (1. + sum(out[j]['min_p'] <= rec['min_p'] for j in null)) / (1. + len(null))
    return out


def false_positive_rates(pvals, keys, alpha):
    names = sorted(next(iter(pvals.values()))['p'])
    rates = {s: float(np.nanmean([pvals[k]['p'][s] <= alpha if np.isfinite(pvals[k]['p'][s]) else np.nan for k in keys]))
             for s in names}
    out = {'subjects': len(keys), 'nominal_alpha': alpha, 'per_score': rates,
           'primary_holm_any': float(np.mean([pvals[k]['holm_any'] for k in keys])),
           'primary_family_minp': float(np.mean([pvals[k]['p_family'] <= alpha for k in keys]))}
    elements = {}
    for pop_key in pvals[keys[0]]['elements']:
        elements[pop_key] = {f: {'subjects_with_discovery': float(np.mean([pvals[k]['elements'][pop_key][f]['reject'].any() for k in keys])),
                                 'mean_discoveries': float(np.mean([pvals[k]['elements'][pop_key][f]['reject'].sum() for k in keys]))}
                             for f in ELEMENT_FAMILIES}
        elements[pop_key]['any_family'] = float(np.mean([any(pvals[k]['elements'][pop_key][f]['reject'].any() for f in ELEMENT_FAMILIES) for k in keys]))
    out['elements_bh'] = elements
    population = {}
    for pop_key in pvals[keys[0]]['elements']:
        fam = lambda k, f: pvals[k]['elements'][pop_key][f]
        population[pop_key] = {f: {'subjects_with_discovery': float(np.mean([fam(k, f)['reject_population'].any() for k in keys])),
                                   'mean_discoveries': float(np.mean([fam(k, f)['reject_population'].sum() for k in keys])),
                                   'element_fraction_p_le_alpha': float(np.nanmean(np.concatenate([fam(k, f)['p_population'] <= alpha
                                                                                                    for k in keys]).astype(float)))
                                   if any(len(fam(k, f)['p_population']) for k in keys) else None}
                               for f in ELEMENT_FAMILIES}
        population[pop_key]['any_family'] = float(np.mean([any(fam(k, f)['reject_population'].any() for f in ELEMENT_FAMILIES) for k in keys]))
    out['elements_bh_population'] = population
    return out


# ---------------------------------------------------------------- calibration selection
def select_calibration(method, wname, sigma, cal_keys, smooth_mm, methods, max_voxels, seed=0):
    """Pooled ROI voxels of the calibration queries (LOO maps): best (smoothing, recalibration) by log-likelihood."""
    S = STATE
    n_vox = int(np.prod(S['shape'])); rng = np.random.default_rng(seed)
    maps, labels, rois, effs = [], [], [], []
    frame_method = 'learned' if method == 'decoder' else method
    for key in cal_keys:
        q = S['queries'][key]; exclude = S['train_index'][q['subject_id']]
        w = kernel_weights(S['ages'], q['age'], sigma, exclude)
        p = S['stacks'][method].full(S['stacks'][method].values(w, uniform_exclude=exclude if sigma is None else None), n_vox)
        m = q['methods'][frame_method]
        y = np.unpackbits(m['y'], count=n_vox).astype(bool); valid = np.unpackbits(m['valid'], count=n_vox).astype(bool)
        roi = np.flatnonzero(ndi.binary_dilation(S['support_mask'] | y, iterations=S['roi_dilation']) & valid)
        maps.append(p); labels.append(y); rois.append(roi); effs.append(ess(w))
    n_eff = float(np.median(effs))
    total = sum(len(r) for r in rois)
    keep = [np.sort(rng.choice(r, max(1, int(len(r) * min(1., max_voxels / total))), replace=False)) for r in rois]
    rows, best = [], None
    for sig in smooth_mm:
        pf = np.concatenate([(ndi.gaussian_filter(p.reshape(S['shape']), sig).ravel() if sig > 0 else p)[k] for p, k in zip(maps, keep)])
        yf = np.concatenate([y[k] for y, k in zip(labels, keep)])
        for name in methods:
            rc = Recalibrator(name, n_eff).fit(pf, yf) if name != 'none' else Recalibrator('none', n_eff)
            ll = calibration(rc(pf), yf, np.ones(len(pf), bool), n_eff)['loglik_per_voxel']
            rows.append({'sigma_mm': sig, 'method': name, 'loglik': ll, 'parameters': rc.describe()})
            if best is None or ll > best[0]:
                best = (ll, sig, rc, name)
    return (best[1], best[2]), {'selected': {'sigma_mm': best[1], 'method': best[3]}, 'grid': rows,
                                'voxels_used': int(sum(len(k) for k in keep)), 'calibration_subjects': len(cal_keys)}


# ---------------------------------------------------------------- main
def load_fitted(directory, set_name):
    out = {}
    for path in sorted(Path(directory).glob(f'fitted_{set_name}_*.pt')):
        f = torch.load(path, map_location='cpu', weights_only=False)
        for k, s in enumerate(f['subject_ids']):
            out[s] = (np.asarray(f['A'][k], float), np.asarray(f['t_mm'][k], float))
    return out


def load_decodings(directory, subject_ids, shape, affine):
    packed = {}
    for path in sorted(Path(directory).glob('subjects_*.npz')):
        z = np.load(path)
        if tuple(int(x) for x in z['shape']) != tuple(shape) or not np.allclose(z['affine'], affine):
            raise ValueError(f'{path}: decoding grid differs from the template grid')
        for s, x in zip(z['subject_ids'], z['packed']):
            packed[str(s)] = x
    missing = [s for s in subject_ids if s not in packed]
    if missing:
        raise ValueError(f'{len(missing)} training subjects without a saved decoding, e.g. {missing[0]}')
    return [packed[s] for s in subject_ids]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--sets', required=True)
    p.add_argument('--decodings', required=True); p.add_argument('--refit', required=True); p.add_argument('--output', required=True)
    p.add_argument('--radii-mm', type=float, nargs='+', default=[3.0, 2.0], help='first = primary population')
    p.add_argument('--tau', type=float, default=0.1); p.add_argument('--edge-cover', type=float, default=0.5)
    p.add_argument('--min-skeleton-voxels', type=int, default=5); p.add_argument('--min-radius-mm', type=float, default=1.2)
    p.add_argument('--path-factor', type=float, default=3.0); p.add_argument('--min-limit-mm', type=float, default=2.0)
    p.add_argument('--snap-mm', type=float, default=1.0); p.add_argument('--roi-dilation', type=int, default=3)
    p.add_argument('--age-sigmas', type=float, nargs='*', default=[7.0, 5.0, 10.0], help='first = calibrated age weighting')
    p.add_argument('--smooth-mm', type=float, nargs='+', default=[0., 1., 2., 3.])
    p.add_argument('--recalibration', nargs='+', default=['none', 'platt', 'isotonic'])
    p.add_argument('--calibration-voxels', type=int, default=5_000_000)
    p.add_argument('--alpha', type=float, default=0.05); p.add_argument('--q', type=float, default=0.05)
    p.add_argument('--methods', nargs='+', default=['decoder', 'learned', 'centroid'])
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    started = time.time()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    model = SDFModel(config, len(frame)); model.load_state_dict(checkpoint['model'])
    train_a, train_t = (x.double().numpy() for x in model.maps_mm())
    sets = json.loads(Path(a.sets).read_text())
    train_ids = frame.subject_id.tolist()
    if (set(sets['null']) | set(sets['calibration'])) - set(train_ids) or set(sets['validation']) & set(train_ids):
        raise ValueError('Null/calibration must be training subjects; validation must not')
    refit = {s: load_fitted(a.refit, s) for s in ('null', 'calibration', 'validation')}
    for s in refit:
        if set(refit[s]) != set(sets[s]):
            raise ValueError(f'Refitted {s} subjects differ from the set ({len(refit[s])} vs {len(sets[s])})')
    _, shape, affine = world_grid(config, [1.0] * 3)
    shape = tuple(int(x) for x in shape)
    n_vox = int(np.prod(shape))
    STATE.update(shape=shape, affine=affine, template_mm=template_world(shape, affine), bbox=config['data']['world_bbox_mm'],
                 roi_dilation=a.roi_dilation, ages=frame.scan_age.to_numpy(float),
                 train_index={s: i for i, s in enumerate(train_ids)},
                 ball=lambda r: np.linalg.norm(np.indices((2 * int(np.ceil(r)) + 1,) * 3) - int(np.ceil(r)), axis=0) <= r)
    from analysis.alignment.evaluate_alignment import subject_geometry
    rows_train = [row for _, row in frame.iterrows()]
    with Pool(a.workers) as pool:
        geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for row in rows_train])
    STATE['mean_centroid'] = np.mean([train_a[i] @ (cen - c) + train_t[i] for i, (c, cen, _) in enumerate(geo)], 0)
    graph_methods = [m for m in ('learned', 'centroid') if m in a.methods or (m == 'learned' and 'decoder' in a.methods)]
    # training subjects under each method's TRAINING transforms
    with Pool(a.workers) as pool:
        train_prep = pool.map(subject_preparation, [(row, {'learned': (train_a[i], train_t[i]), 'centroid': (np.eye(3), None)},
                                                     a.min_radius_mm) for i, row in enumerate(rows_train)])
    # queries: null + calibration (training subjects, refitted transforms) and validation
    val_rows = held_out_rows(config, sets['validation'])
    val_ages = dict(zip(val_rows.subject_id, val_rows.scan_age.astype(float)))
    query_rows = [(s, set_name, frame.set_index('subject_id', drop=False).loc[s] if set_name != 'validation'
                   else val_rows.set_index('subject_id', drop=False).loc[s]) for set_name in ('null', 'calibration', 'validation')
                  for s in sets[set_name]]
    with Pool(a.workers) as pool:
        query_prep = pool.map(subject_preparation, [(row, {'learned': refit[set_name][s], 'centroid': (np.eye(3), None)}, a.min_radius_mm)
                                                    for s, set_name, row in query_rows])
    queries = {}
    for (s, set_name, row), prep in zip(query_rows, query_prep):
        queries[f'{set_name}/{s}'] = {'subject_id': s, 'set': set_name, 'row': row,
                                      'age': float(row['scan_age']), 'methods': prep['methods']}
    STATE['queries'] = queries
    # population support: any training subject positive in any decoding or mapped label
    decodings = load_decodings(a.decodings, train_ids, shape, affine) if 'decoder' in a.methods else []
    support_mask = np.zeros(n_vox, bool)
    for x in decodings:
        support_mask |= np.unpackbits(x, count=n_vox).astype(bool)
    for prep in train_prep:
        for m in prep['methods'].values():
            support_mask |= np.unpackbits(m['y'], count=n_vox).astype(bool)
    support = np.flatnonzero(support_mask)
    STATE['support_mask'] = support_mask
    stacks = {}
    if 'decoder' in a.methods:
        stacks['decoder'] = Stack(decodings, None, support)
    for m in ('learned', 'centroid'):
        if m in a.methods:
            stacks[m] = Stack([prep['methods'][m]['y'] for prep in train_prep], [prep['methods'][m]['valid'] for prep in train_prep], support)
    STATE['stacks'] = stacks
    # population graphs and attributes per method
    populations, presences, train_attr, query_attr = {}, {}, {}, {}
    params = {'min_radius_mm': a.min_radius_mm, 'path_factor': a.path_factor, 'min_limit_mm': a.min_limit_mm,
              'snap_mm': a.snap_mm, 'grid_affine': affine, 'grid_shape': np.asarray(shape)}
    STATE['params'] = params
    pop_summary = {}
    for m in graph_methods:
        clouds = [prep['methods'][m]['lines'] for prep in train_prep]
        populations[m], presences[m] = {}, {}
        for r in a.radii_mm:
            pop = build_population(clouds, shape, affine, r, a.tau, a.edge_cover, a.min_skeleton_voxels)
            key = f'r{r:g}mm_tau{a.tau:g}'
            populations[m][key] = {k: pop[k] for k in ('nodes_mm', 'edges', 'edge_points_mm', 'edge_probability', 'radius_mm', 'edge_cover')}
            presences[m][key] = pop['presence'].astype(float)
            pop_summary[f'{m}/{key}'] = {'nodes': len(pop['nodes_mm']), 'edges': len(pop['edges']),
                                         'total_length_mm': float(sum(pop['edge_length_mm'])), 'map_max': float(pop['cmap'].max())}
            save(pop['cmap'], affine, out / f'CentrelineProbability_{m}_r{r:g}mm.nii.gz')
    STATE['populations'] = populations
    STATE['presence'] = presences
    STATE['primary_population'] = f'r{a.radii_mm[0]:g}mm_tau{a.tau:g}'
    with Pool(a.workers) as pool:
        for m in graph_methods:
            train_attr[m] = pool.map(attributes_worker, [(row, train_prep[i]['methods'][m]['A'], train_prep[i]['methods'][m]['t'], m)
                                                         for i, row in enumerate(rows_train)])
            keys = list(queries)
            res = pool.map(attributes_worker, [(queries[k]['row'], queries[k]['methods'][m]['A'], queries[k]['methods'][m]['t'], m) for k in keys])
            query_attr[m] = dict(zip(keys, res))
    STATE['train_attributes'] = train_attr
    STATE['attributes'] = query_attr
    STATE['train_affine'] = np.array([affine_vector(train_a[i], train_t[i]) for i in range(len(train_ids))])
    prep_s = time.time() - started
    # calibration selection on the calibration set (uniform and the first age weighting)
    weightings = [('uniform', None)] + [(f'age_sigma{s:g}', s) for s in a.age_sigmas]
    cal_keys = [k for k in queries if queries[k]['set'] == 'calibration']
    STATE['calibrators'] = {}
    calibration_report = {}
    for method in a.methods:
        for wname, sigma in weightings[:2]:
            chosen, report = select_calibration(method, wname, sigma, cal_keys, a.smooth_mm, a.recalibration, a.calibration_voxels)
            STATE['calibrators'][(method, wname)] = chosen
            calibration_report[f'{method}/{wname}'] = report
            print(json.dumps({'calibration': f'{method}/{wname}', **report['selected']}), flush=True)
    cal_s = time.time() - started - prep_s
    # scoring
    eval_keys = [k for k in queries if queries[k]['set'] in ('null', 'validation')]
    tasks = [(k, m, wn, sg, queries[k]['set'] == 'validation' and wn == 'uniform') for k in eval_keys for m in a.methods for wn, sg in weightings]
    with Pool(a.workers) as pool:
        results = {(k, m, wn): (sc, el) for k, m, wn, sc, el in pool.imap_unordered(score_query, tasks, chunksize=4)}
    score_s = time.time() - started - prep_s - cal_s
    keys_null = [k for k in eval_keys if queries[k]['set'] == 'null']
    keys_val = [k for k in eval_keys if queries[k]['set'] == 'validation']
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'], 'settings': vars(a),
               'sets': {k: len(v) for k, v in sets.items() if isinstance(v, list)},
               'support_voxels': int(len(support)), 'populations': pop_summary, 'calibration': calibration_report,
               'notes': {'null': 'training subjects with test-time refitted transforms, leave-one-out (own label, graph, '
                                 'decoding and trained transform removed); the network itself was trained on them',
                         'centroid': 'A = I; scaled radius = native radius',
                         'p_values': '(1 + #null >= s) / (1 + #null); null queries compared with the other null queries',
                         'maps': 'per-location maps saved for validation subjects (uniform weighting) only'},
               'fpr': {}, 'score_medians': {}, 'defects': {}, 'ess': {}}
    tables = []
    for m in a.methods:
        for wn, _ in weightings:
            pv = null_pvalues(results, m, wn, keys_null, keys_val, a.alpha, a.q)
            summary['fpr'][f'{m}/{wn}'] = {'null_loo': false_positive_rates(pv, keys_null, a.alpha),
                                          'validation': false_positive_rates(pv, keys_val, a.alpha)}
            names = sorted(results[(keys_null[0], m, wn)][0])
            summary['score_medians'][f'{m}/{wn}'] = {
                g: {s: float(np.nanmedian([results[(k, m, wn)][0].get(s, np.nan) for k in ks])) for s in names}
                for g, ks in (('null', keys_null), ('validation', keys_val))}
            summary['ess'][f'{m}/{wn}'] = {g: [float(np.min(e)), float(np.median(e))] for g, ks in (('null', keys_null), ('validation', keys_val))
                                           for e in [[results[(k, m, wn)][0]['ess'] for k in ks]]}
            for k in keys_null + keys_val:
                sc, el = results[(k, m, wn)]
                for s, v in sc.items():
                    tables.append({'subject_id': queries[k]['subject_id'], 'set': queries[k]['set'], 'method': m, 'weighting': wn,
                                   'score': s, 'value': v, 'p': pv[k]['p'].get(s, np.nan)})
                prim = STATE['primary_population']
                if prim in el and 'defects' in el[prim] and wn == 'uniform':
                    for label in el[prim]['defects']:
                        summary['defects'].setdefault(f'{m}/{queries[k]["set"]}', {}).setdefault(label, 0)
                        summary['defects'][f'{m}/{queries[k]["set"]}'][label] += 1
                if wn == 'uniform' or wn == weightings[1][0]:
                    write_subject_report(out, queries[k], m, wn, sc, el, pv[k], populations.get(m, {}), affine, shape)
    pd.DataFrame(tables).to_csv(out / 'scores.tsv', sep='\t', index=False)
    summary['runtime_s'] = {'preparation': prep_s, 'calibration': cal_s, 'scoring': score_s, 'total': time.time() - started}
    (out / 'deviation.json').write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    print(json.dumps({k: {g: v[g]['primary_holm_any'] for g in v} for k, v in summary['fpr'].items()}, indent=1))


def write_subject_report(out, query, method, wname, scores, elements, pv, pops, affine, shape):
    """Per-subject deviation report: scores with empirical p, per-edge and per-node tables; maps for validation."""
    directory = out / 'subjects' / query['subject_id']
    directory.mkdir(parents=True, exist_ok=True)
    report = {'subject_id': query['subject_id'], 'set': query['set'], 'age': query['age'], 'method': method, 'weighting': wname,
              'scores': scores, 'p': pv['p'], 'primary_scores': pv['primary'], 'flagged_holm': pv['holm_any'],
              'p_family_minp': pv['p_family'], 'populations': {}}
    for pop_key, el in elements.items():
        if not pop_key.startswith('r'):
            continue
        fam = pv['elements'][pop_key]
        pop = pops[pop_key]
        report['populations'][pop_key] = {
            'edges': [{'edge': list(pop['edges'][i]), 'p_edge': float(el['edge_p'][i]), 'present': bool(el['edge_present'][i]),
                       'surprise': float(el['edge_surprise'][i]), 'p_emp': float(fam['edge_surprise']['p'][i]),
                       'bh_reject': bool(fam['edge_surprise']['reject'][i]),
                       'p_population': float(fam['edge_surprise']['p_population'][i]),
                       'bh_reject_population': bool(fam['edge_surprise']['reject_population'][i]),
                       'conn_graph_surprise': float(el['conn_graph'][i]), 'conn_graph_p_emp': float(fam['conn_graph']['p'][i]),
                       'conn_graph_bh_reject': bool(fam['conn_graph']['reject'][i]),
                       'conn_graph_p_population': float(fam['conn_graph']['p_population'][i]),
                       'conn_graph_bh_reject_population': bool(fam['conn_graph']['reject_population'][i]),
                       'conn_mask_surprise': float(el['conn_mask'][i]), 'conn_mask_p_emp': float(fam['conn_mask']['p'][i]),
                       'conn_mask_bh_reject': bool(fam['conn_mask']['reject'][i]),
                       'conn_mask_p_population': float(fam['conn_mask']['p_population'][i]),
                       'conn_mask_bh_reject_population': bool(fam['conn_mask']['reject_population'][i])} for i in range(len(pop['edges']))],
            'nodes': [{'node': i, 'position_mm': np.asarray(pop['nodes_mm'][i]).tolist(), 'radius_z': float(el['radius_z'][i]),
                       'radius_percentile': float(el['radius_percentile'][i]), 'floor': bool(el['floor'][i]),
                       'p_emp': float(fam['radius_absz']['p'][i]), 'bh_reject': bool(fam['radius_absz']['reject'][i]),
                       'p_population': float(fam['radius_absz']['p_population'][i]),
                       'bh_reject_population': bool(fam['radius_absz']['reject_population'][i])}
                      for i in range(len(pop['nodes_mm']))],
            'absent_expected_edges': el.get('defects')}
        if 'voxel_surprise_map' in elements:  # validation, uniform: paint per-edge / per-node values
            edge_map = np.zeros(shape, np.float32); node_map = np.zeros(shape, np.float32)
            for i, path in enumerate(pop['edge_points_mm']):
                edge_map[within_radius(path, shape, affine, 1.0)] = el['edge_surprise'][i]
            for i, x in enumerate(pop['nodes_mm']):
                if np.isfinite(el['radius_z'][i]):
                    node_map[within_radius(np.asarray(x)[None], shape, affine, 1.5)] = el['radius_z'][i]
            save(edge_map, affine, directory / f'EdgeSurprise_{method}_{pop_key}.nii.gz')
            save(node_map, affine, directory / f'NodeRadiusZ_{method}_{pop_key}.nii.gz')
    if 'voxel_surprise_map' in elements:
        save(elements['voxel_surprise_map'].reshape(shape), affine, directory / f'VoxelSurprise_{method}.nii.gz')
    (directory / f'report_{method}_{wname}.json').write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


if __name__ == '__main__':
    main()
