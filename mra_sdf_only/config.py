from copy import deepcopy
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
    for key in ('mra', 'sdf'):
        s = c['sampling'][key]
        if isinstance(s['points'], bool) or not isinstance(s['points'], int) or s['points'] <= 0:
            raise ValueError('Each task requires a positive integer point count')
        names = {'uniform', 'bright'} if key == 'mra' else {'surface', 'interior', 'outside'}
        f = s['fractions']
        if set(f) != names or not all(np.isfinite(v) and v >= 0 for v in f.values()) or not np.isclose(sum(f.values()), 1):
            raise ValueError(f'Invalid {key} fractions')
    if not 0 <= c['sampling']['mra']['bright_quantile'] <= 1 or c['sampling']['sdf']['surface_band_mm'] <= 0:
        raise ValueError('Invalid sampling band or quantile')
    for weight in ('mra_weight', 'sdf_weight', 'moe_weight', 'tf_weight'):
        if not np.isfinite(c['loss'][weight]) or c['loss'][weight] < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    if c['model']['moe_k'] < 1 or c['model']['moe_k'] > c['model']['num_experts']:
        raise ValueError('Invalid top-k routing')
    if c['model']['mra_activation'] not in ('linear', 'sigmoid'):
        raise ValueError('Invalid MRA activation')
    for size in c['data']['world_bbox_mm'] + c['atlas']['spacing_mm']:
        if not np.isfinite(size) or size <= 0:
            raise ValueError('Bounding box and spacing must be positive')
    if not 0 < c['data']['intensity_percentile'] <= 100:
        raise ValueError('Intensity percentile must be in (0,100]')
    if c['data']['age_range'][1] <= c['data']['age_range'][0]:
        raise ValueError('Invalid age range')
    for key in ('subjects_per_rank', 'task_points_per_step', 'epochs', 'checkpoint_every'):
        if c['training'][key] <= 0:
            raise ValueError(f'Invalid training.{key}')
    if c['training']['atlas_every'] < 0 or c['optimizer']['scheduler_epochs'] <= 0:
        raise ValueError('Invalid atlas period or scheduler horizon')


def condition_vector(config, age, sex):
    lo, hi = config['data']['age_range']
    categories = config['data']['sex_categories']
    if sex not in categories:
        raise ValueError(f'Unknown sex {sex!r}; expected {categories}')
    scale = config['data']['condition_scale']
    return np.asarray([(2 * (float(age)-lo)/(hi-lo)-1)*scale,
                       (2 * categories.index(sex)/max(1,len(categories)-1)-1)*scale], dtype=np.float32)
