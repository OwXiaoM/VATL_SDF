from pathlib import Path
import yaml
import numpy as np


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    for assignment in overrides:
        key, value = assignment.split('=', 1)
        obj = config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                raise ValueError(f'Unknown configuration section: {key}')
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError(f'Unknown configuration key: {key}')
        obj[parts[-1]] = yaml.safe_load(value)
    validate(config)
    return config


def validate(c):
    if c['sdf']['sign_convention'] != 'negative_inside' or c['sdf']['distance_units'] != 'mm':
        raise ValueError('Only negative-inside SDF targets in mm are supported')
    if not np.isfinite(c['sdf']['truncation_mm']) or c['sdf']['truncation_mm'] <= 0:
        raise ValueError('SDF truncation must be positive and finite')
    s = c['sampling']['sdf']
    if isinstance(s['points'], bool) or not isinstance(s['points'], int) or s['points'] <= 0:
        raise ValueError('sampling.sdf.points must be a positive integer')
    f = s['fractions']
    if set(f) != {'surface', 'interior', 'outside'} or not all(np.isfinite(v) and v >= 0 for v in f.values()) \
            or not np.isclose(sum(f.values()), 1):
        raise ValueError('Invalid sdf fractions')
    if s['surface_band_mm'] <= 0:
        raise ValueError('Invalid surface band')
    for weight in ('sdf_weight', 'moe_weight', 'tf_weight'):
        if not np.isfinite(c['loss'][weight]) or c['loss'][weight] < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    if c['model']['moe_k'] < 1 or c['model']['moe_k'] > c['model']['num_experts']:
        raise ValueError('Invalid top-k routing')
    for size in c['data']['world_bbox_mm'] + c['atlas']['spacing_mm']:
        if not np.isfinite(size) or size <= 0:
            raise ValueError('Bounding box and spacing must be positive')
    if c['data']['age_range'][1] <= c['data']['age_range'][0]:
        raise ValueError('Invalid age range')
    for key in ('subjects_per_rank', 'task_points_per_step', 'epochs', 'checkpoint_every'):
        if c['training'][key] <= 0:
            raise ValueError(f'Invalid training.{key}')
    horizon = c['optimizer']['scheduler_epochs']
    if c['training']['atlas_every'] < 0 or not (horizon == 'run' or (isinstance(horizon, int) and horizon > 0)):
        raise ValueError("Invalid atlas period or scheduler horizon (positive int or 'run')")
    validate_transform(c)
    validate_alignment(c)
    validate_graph(c.get('graph', {}))
    fields = c.get('graph_fields', {})
    for name in field_names(c, enabled_only=False):
        if not isinstance(fields[name]['enabled'], bool) or fields[name]['weight'] < 0:
            raise ValueError(f'Invalid graph_fields.{name}')
    if fields and min(fields['band_mm'], fields['junction_sigma_mm'], fields['short_loop_mm'], fields['exclusion_mm']) <= 0:
        raise ValueError('Invalid graph_fields geometry settings')


def validate_transform(c):
    t = c['model'].get('transform', {})
    if not t:
        return
    if t.get('mode', 'legacy') not in ('legacy', 'normalized') or t.get('type', 'rigid') not in ('rigid', 'affine'):
        raise ValueError('model.transform.mode must be legacy|normalized and type rigid|affine')
    if t.get('mode', 'legacy') == 'legacy':
        if t.get('type', 'rigid') != 'rigid' or t.get('mean_identity') or t.get('centroid_init'):
            raise ValueError('legacy transforms are rigid, without mean_identity or centroid_init')
        return
    for key in ('lr_rotation', 'lr_translation') + (('lr_scale',) if t['type'] == 'affine' else ()):
        if not c['optimizer'].get(key, 0) > 0:
            raise ValueError(f'normalized transforms need optimizer.{key} > 0')
    if c['loss']['tf_weight'] != 0:
        raise ValueError('loss.tf_weight applies to legacy transforms only; use model.transform.affine_reg_weight')
    if t.get('affine_reg_weight', 0) < 0:
        raise ValueError('affine_reg_weight must be nonnegative')


def validate_alignment(c):
    a = c.get('alignment', {})
    if not a or not a.get('enabled'):
        return
    if not field_names(c) and not c.get('graph_fields'):
        raise ValueError('alignment needs the graph_fields section (centreline construction settings)')
    if not c.get('graph'):
        raise ValueError('alignment needs the graph section (mask support)')
    if not (isinstance(a['epochs'], int) and a['epochs'] > 0):
        raise ValueError('alignment.epochs must be a positive integer')
    for key in ('sdf_truncation_mm', 'large_vessel_min_radius_mm'):
        if not a[key] > 0:
            raise ValueError(f'alignment.{key} must be positive')
    for key in ('sdf_weight', 'large_vessel_weight', 'latent_prior_weight'):
        if not a[key] >= 0:
            raise ValueError(f'alignment.{key} must be nonnegative')


GRAPH_POINT_TERMS = ('centerline_sdf', 'centerline_hinge', 'edge_hinge', 'radius')


def validate_graph(g):
    if not g:
        return
    if g['support_tolerance_mm'] < 0 or g['edge_step_mm'] <= 0 or not isinstance(g['support_strict'], bool):
        raise ValueError('Invalid graph support settings')
    for name in GRAPH_POINT_TERMS + ('eikonal',):
        term = g[name]
        if not isinstance(term['enabled'], bool) or not np.isfinite(term['weight']) or term['weight'] < 0:
            raise ValueError(f'Invalid graph.{name}')
        if name in GRAPH_POINT_TERMS and not 0 < term['points_fraction'] <= 1:
            raise ValueError(f'graph.{name}.points_fraction must be in (0, 1]')
    if g['eikonal']['band_mm'] <= 0 or g['radius']['min_radius_mm'] < 0:
        raise ValueError('Invalid eikonal band or radius threshold')


FIELD_NAMES = ('centerline_distance', 'radius', 'tangent', 'junction')


def field_names(config, enabled_only=True):
    """Phase 4 implicit graph-field heads (enabled ones by default)."""
    fields = config.get('graph_fields', {})
    return [n for n in FIELD_NAMES if fields and (not enabled_only or fields[n]['enabled'])]


def graph_terms(config):
    """Names of enabled graph point terms (each draws its own extra points per subject)."""
    g = config.get('graph', {})
    return [name for name in GRAPH_POINT_TERMS if g and g[name]['enabled']]


def condition_vector(config, age, sex):
    lo, hi = config['data']['age_range']
    categories = config['data']['sex_categories']
    if sex not in categories:
        raise ValueError(f'Unknown sex {sex!r}; expected {categories}')
    scale = config['data']['condition_scale']
    return np.asarray([(2 * (float(age)-lo)/(hi-lo)-1)*scale,
                       (2 * categories.index(sex)/max(1,len(categories)-1)-1)*scale], dtype=np.float32)
