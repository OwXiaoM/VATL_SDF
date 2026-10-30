"""Authoritative ordered subject splits, shared by population training and evaluation."""
from itertools import combinations
from pathlib import Path
import yaml


def read_splits(path, key, expected_counts=None):
    if not path:
        raise ValueError('Explicit subject split YAML is required; training on all TSV rows is forbidden')
    value = yaml.safe_load(Path(path).read_text())
    for part in (key or '').split('.') if key else []:
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f'Missing split YAML key: {key}')
        value = value[part]
    if not isinstance(value, dict) or set(value) != {'train', 'val', 'test'}:
        raise ValueError('Split node must contain train, val, and test lists')
    for name, ids in value.items():
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError(f'{name} must be a list of subject ID strings')
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate subject IDs in {name} split')
    for a, b in combinations(value, 2):
        overlap = set(value[a]) & set(value[b])
        if overlap:
            raise ValueError(f'{a}/{b} split overlap: {sorted(overlap)[:5]}')
    if expected_counts is not None:
        if {name: len(ids) for name, ids in value.items()} != expected_counts:
            raise ValueError(f'Split sizes differ from expected counts {expected_counts}')
    return value


def ordered_rows(frame, ids):
    if frame.subject_id.duplicated().any():
        raise ValueError('Duplicate subject IDs in metadata')
    missing = set(ids)-set(frame.subject_id)
    if missing:
        raise ValueError(f'Split subjects missing from metadata: {sorted(missing)[:5]}')
    return frame.set_index('subject_id', drop=False).loc[ids].reset_index(drop=True).copy()


def check_resume_subjects(checkpoint, frame):
    ids = [row['subject_id'] for row in checkpoint['metadata']]
    if ids != frame.subject_id.tolist():
        raise ValueError('Checkpoint training subjects/order differ from declared train split; contaminated or incompatible resume forbidden')
