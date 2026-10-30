"""Only subject ID, SDF, vessel graph, age and sex are consumed by this loader.

SDF voxels and graph nodes are mapped to normalized coordinates by the same function
(graph_coords.normalize_world with the SDF affine), so both live in one coordinate system.
The binary vessel mask is SDF < 0, which equals segmentation label 1 exactly (Phase 0a).
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .config import condition_vector
from .graph_coords import load_graph, normalize_world, voxel_to_world
from .splits import read_splits, ordered_rows

REQUIRED_COLUMNS = ('subject_id', 'SDF', 'Graph', 'scan_age', 'sex')


def read_metadata(config):
    frame = pd.read_csv(config['data']['tsv'], sep='\t', dtype={'subject_id':str})
    missing = set(REQUIRED_COLUMNS)-set(frame)
    if missing:
        raise ValueError(f'Missing required metadata columns: {sorted(missing)}')
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    if frame.isna().any().any() or frame['subject_id'].duplicated().any():
        raise ValueError('Metadata contains missing values or duplicate subject IDs')
    frame['scan_age'] = pd.to_numeric(frame['scan_age'], errors='raise')
    data = config['data']
    if data.get('split') != 'train':
        raise ValueError('Population data loader requires data.split=train explicitly')
    splits = read_splits(data.get('subject_ids'), data.get('split_key'), data.get('expected_split_counts'))
    declared = set().union(*(set(ids) for ids in splits.values()))
    if data.get('require_complete_split', True) and declared != set(frame.subject_id):
        raise ValueError('Split union must match metadata IDs exactly; unassigned or missing subjects')
    frame = ordered_rows(frame, splits['train'])
    lo, hi = data['age_range']
    if not frame['scan_age'].between(lo, hi).all():
        raise ValueError('Declared training subject age outside configured range; refusing silent filtering')
    if data.get('max_subjects'):
        if config['training'].get('max_steps') is None:
            raise ValueError('max_subjects is allowed only for explicit max_steps smoke checks')
        frame = frame.head(int(data['max_subjects']))
    if frame.empty:
        raise ValueError('No declared training subjects remain')
    parent = Path(config['data']['tsv']).resolve().parent
    for key in ('SDF', 'Graph'):
        frame[key] = frame[key].map(lambda path: str((parent / path).resolve()))
        for path in frame[key]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    for _,row in frame.iterrows():
        condition_vector(config,row.scan_age,row.sex)
    return frame.reset_index(drop=True)


def voxel_coordinates(indices, shape, affine, config):
    """Normalized coordinates of flat voxel indices (voxel centres)."""
    voxels = np.column_stack(np.unravel_index(indices, shape))
    return normalize_world(voxel_to_world(affine, voxels), affine, shape, config['data']['world_bbox_mm']).astype(np.float32)


def load_subject(row, config):
    image = nib.load(row['SDF'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    if sdf.ndim == 4 and sdf.shape[-1] == 1:
        sdf = sdf[..., 0]
    if sdf.ndim != 3:
        raise ValueError(f'{row["subject_id"]}: SDF is not 3D')
    affine = image.affine.copy()
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-9:
        raise ValueError('Invalid spatial affine')
    unit = image.header.get_xyzt_units()[0]
    if unit not in ('mm','unknown'):
        raise ValueError(f'Affine must use mm (or explicitly assumed unknown), got {unit}')
    if not np.isfinite(sdf).all():
        raise ValueError('SDF must contain only finite values')
    if not (np.any(sdf < 0) and np.any(sdf > 0)):
        raise ValueError('Expected both interior-negative and exterior-positive distances; verify SDF preprocessing')
    # Crop only by physical geometry, never by image values or anatomy.
    shape = sdf.shape
    half = np.asarray(config['data']['world_bbox_mm'])/2.
    centre = voxel_to_world(affine, np.asarray(shape)/2.)
    valid = []
    for start in range(0,sdf.size,262144):
        indices = np.arange(start,min(start+262144,sdf.size))
        voxels = np.column_stack(np.unravel_index(indices,shape))
        valid.append(indices[np.all(np.abs((voxel_to_world(affine,voxels)-centre)/half)<=1.,axis=1)])
    valid = np.concatenate(valid)
    if not valid.size:
        raise ValueError('No image voxels inside the physical bounding box')
    graph = load_graph(row['Graph'])
    graph['coords'] = normalize_world(graph['pos'], affine, shape, config['data']['world_bbox_mm']).astype(np.float32)
    report = {'subject_id':str(row['subject_id']), 'shape':list(shape),
              'spacing_mm':nib.affines.voxel_sizes(affine).tolist(), 'header_units':unit,
              'sdf_range_mm':[float(sdf.min()),float(sdf.max())],
              'sdf_negative_voxels':int((sdf<0).sum()),'valid_voxels':int(valid.size),
              'graph_nodes':len(graph['pos']), 'graph_edges':len(graph['edges']),
              'graph_nodes_in_bbox':int((np.abs(graph['coords'])<=1).all(1).sum())}
    return {'sdf':sdf,'affine':affine,'valid':valid,'graph':graph,'report':report}


def sample_pools(pools, fractions, count, rng):
    names = list(fractions)
    weights = np.asarray([fractions[k] if len(pools[k]) else 0 for k in names],dtype=float)
    if weights.sum() <= 0:
        raise ValueError('All requested sampling pools are empty')
    weights /= weights.sum()
    quota = np.floor(weights*count).astype(int)
    order = np.argsort(-(weights*count-quota),kind='stable')
    quota[order[:count-int(quota.sum())]] += 1
    draws, stats = [], {}
    for name, n in zip(names,quota):
        pool = pools[name]
        replacement = len(pool) < n
        chosen = rng.choice(pool,size=n,replace=replacement) if n else np.empty(0,dtype=np.int64)
        draws.append(chosen)
        stats[name] = {'pool_voxels':int(len(pool)), 'draws':int(n), 'replacement':bool(replacement),
                       'requested_fraction':float(fractions[name])}
    indices = np.concatenate(draws)
    rng.shuffle(indices)
    return indices, {'pools':stats,'count':int(count), 'unique_voxels':int(np.unique(indices).size),
                     'duplicate_draws':int(count-np.unique(indices).size),
                     'empty_pools':[k for k in names if not len(pools[k])]}


def sample_sdf(subject, config, rng):
    valid = subject['valid']; sdf = subject['sdf'].ravel()
    band = config['sampling']['sdf']['surface_band_mm']
    surface = valid[np.abs(sdf[valid])<=band]
    interior = valid[sdf[valid]<0]
    outside = valid[sdf[valid]>band]
    si, sr = sample_pools({'surface':surface,'interior':interior,'outside':outside},
                          config['sampling']['sdf']['fractions'],config['sampling']['sdf']['points'],rng)
    sr['surface_interior_overlap_voxels'] = int(((sdf[valid]<0)&(np.abs(sdf[valid])<=band)).sum())
    sr['outside_definition'] = 'SDF > surface_band_mm; surface/interior intentionally overlap'
    sr['sampled_negative_fraction'] = float((sdf[si]<0).mean())
    sr['sampled_surface_fraction'] = float((np.abs(sdf[si])<=band).mean())
    return {'sdf_coords':voxel_coordinates(si,subject['sdf'].shape,subject['affine'],config),
            'sdf_target_mm':sdf[si,None].copy(), 'sampling_report':{'sdf':sr}}


class SDFGraphDataset(Dataset):
    def __init__(self, config, frame=None):
        self.config = config
        declared = read_metadata(config)
        if frame is not None and (frame.subject_id.duplicated().any() or not set(frame.subject_id).issubset(set(declared.subject_id))):
            raise ValueError('Explicit dataset frame contains subjects outside the declared population train split')
        self.frame = declared if frame is None else frame.reset_index(drop=True).copy()
        self.epoch = 0
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.frame)

    def subject(self, index):
        if index not in self.cache:
            subject = load_subject(self.frame.iloc[index],self.config)
            if self.config['data']['cache_subjects'] > 0:
                self.cache[index] = subject
                while len(self.cache)>self.config['data']['cache_subjects']:
                    self.cache.popitem(last=False)
            return subject
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        subject = self.subject(index)
        # Independent of worker scheduling/rank; fully reproducible at epoch boundaries.
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index]))
        result = sample_sdf(subject,self.config,rng)
        result['sampling_report'].update({'subject_index':int(index),'epoch':int(self.epoch)})
        result['condition'] = condition_vector(self.config,self.frame.iloc[index].scan_age,self.frame.iloc[index].sex)
        result['subject_index'] = int(index)
        return result


def collate_subjects(items):
    result = {k:torch.from_numpy(np.stack([item[k] for item in items])) for k in ('sdf_coords','sdf_target_mm','condition')}
    result['subject_index'] = torch.tensor([item['subject_index'] for item in items],dtype=torch.long)
    result['sampling_report'] = [item['sampling_report'] for item in items]
    return result
