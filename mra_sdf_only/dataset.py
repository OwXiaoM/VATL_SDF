"""Only MRA, SDF, age, sex and subject ID are consumed by this loader."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .config import condition_vector
from .splits import read_splits, ordered_rows

REQUIRED_COLUMNS = ('subject_id', 'MRA', 'SDF', 'scan_age', 'sex')


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
    for key in ('MRA','SDF'):
        frame[key] = frame[key].map(lambda path: str((parent / path).resolve()))
        for path in frame[key]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    for _,row in frame.iterrows():
        condition_vector(config,row.scan_age,row.sex)
    return frame.reset_index(drop=True)


def normalize_mra(mra, config):
    mra = np.asarray(mra,dtype=np.float32)
    original_min,original_max = float(mra.min()),float(mra.max())
    mode = config['data']['intensity_normalization']
    if mode == 'preserve_unit_or_percentile':
        if original_min >= 0 and original_max <= 1:
            scale = 1.
        else:
            scale = float(np.percentile(np.maximum(mra,0),config['data']['intensity_percentile']))
            if scale <= 0:
                scale = original_max
            if scale <= 0:
                raise ValueError('Nonpositive MRA normalization scale')
        mra = np.clip(mra / scale,0,1)
    elif mode == 'unit':
        if original_min < 0 or original_max > 1:
            raise ValueError('MRA is outside [0,1] with unit normalization')
        scale = 1.
    else:
        raise ValueError('Unknown intensity normalization mode')
    return mra,scale


def load_pair(row, config):
    images = [nib.load(row[key]) for key in ('MRA','SDF')]
    arrays = [np.asarray(image.dataobj,dtype=np.float32) for image in images]
    arrays = [a[...,0] if a.ndim==4 and a.shape[-1]==1 else a for a in arrays]
    if any(a.ndim != 3 for a in arrays) or arrays[0].shape != arrays[1].shape:
        raise ValueError(f'{row["subject_id"]}: MRA/SDF shapes disagree or are not 3D')
    if not np.allclose(images[0].affine,images[1].affine,rtol=1e-5,atol=1e-3):
        raise ValueError(f'{row["subject_id"]}: MRA/SDF affines disagree')
    affine = images[0].affine.copy()
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-9:
        raise ValueError('Invalid spatial affine')
    for image in images:
        unit = image.header.get_xyzt_units()[0]
        if unit not in ('mm','unknown'):
            raise ValueError(f'Affine must use mm (or explicitly assumed unknown), got {unit}')
    mra, sdf = arrays
    if not np.isfinite(mra).all() or not np.isfinite(sdf).all():
        raise ValueError('MRA and SDF must contain only finite values')
    if not (np.any(sdf < 0) and np.any(sdf > 0)):
        raise ValueError('Expected both interior-negative and exterior-positive distances; verify SDF preprocessing')
    original_min,original_max = float(mra.min()),float(mra.max())
    mra,scale = normalize_mra(mra,config)
    # Crop only by physical geometry, never by image intensity or anatomy.
    shape = mra.shape
    center = nib.affines.apply_affine(affine,np.asarray(shape)/2.)
    valid = []
    half = np.asarray(config['data']['world_bbox_mm'])/2.
    for start in range(0,mra.size,262144):
        indices = np.arange(start,min(start+262144,mra.size))
        voxels = np.column_stack(np.unravel_index(indices,shape))
        coords = (nib.affines.apply_affine(affine,voxels)-center)/half
        valid.append(indices[np.all(np.abs(coords)<=1.,axis=1)])
    valid = np.concatenate(valid)
    if not valid.size:
        raise ValueError('No image voxels inside the physical bounding box')
    report = {'subject_id':str(row['subject_id']), 'shape':list(shape),
              'spacing_mm':nib.affines.voxel_sizes(affine).tolist(),
              'header_units':[image.header.get_xyzt_units()[0] for image in images],
              'mra_original_range':[original_min,original_max], 'mra_scale':scale,
              'mra_exact_zero_fraction':float((arrays[0]==0).mean()),
              'sdf_range_mm':[float(sdf.min()),float(sdf.max())],
              'sdf_negative_voxels':int((sdf<0).sum()),'valid_voxels':int(valid.size),
              'sdf_units_assumed_mm':True,
              'sign_semantics':'negative_inside declared; cannot prove from a distance field alone'}
    return {'mra':mra,'sdf':sdf,'affine':affine,'valid':valid,'report':report}


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


def sample_tasks(pair, config, rng):
    valid = pair['valid']; mra = pair['mra'].ravel(); sdf = pair['sdf'].ravel()
    positives = mra[valid][mra[valid]>0]
    threshold = float(np.quantile(positives,config['sampling']['mra']['bright_quantile'])) if positives.size else float('inf')
    bright = valid[(mra[valid]>0)&(mra[valid]>=threshold)]
    mi, mr = sample_pools({'uniform':valid,'bright':bright},config['sampling']['mra']['fractions'],config['sampling']['mra']['points'],rng)
    band = config['sampling']['sdf']['surface_band_mm']
    surface = valid[np.abs(sdf[valid])<=band]
    interior = valid[sdf[valid]<0]
    outside = valid[sdf[valid]>band]
    si, sr = sample_pools({'surface':surface,'interior':interior,'outside':outside},config['sampling']['sdf']['fractions'],config['sampling']['sdf']['points'],rng)
    mr['bright_threshold'] = threshold if np.isfinite(threshold) else None
    mr['uniform_bright_overlap_voxels'] = int(bright.size)
    sr['surface_interior_overlap_voxels'] = int(((sdf[valid]<0)&(np.abs(sdf[valid])<=band)).sum())
    sr['outside_definition'] = 'SDF > surface_band_mm; surface/interior intentionally overlap'
    sr['sampled_negative_fraction'] = float((sdf[si]<0).mean())
    sr['sampled_surface_fraction'] = float((np.abs(sdf[si])<=band).mean())
    def coordinates(indices):
        voxels = np.column_stack(np.unravel_index(indices,pair['mra'].shape))
        center = nib.affines.apply_affine(pair['affine'],np.asarray(pair['mra'].shape)/2.)
        return ((nib.affines.apply_affine(pair['affine'],voxels)-center)/(np.asarray(config['data']['world_bbox_mm'])/2.)).astype(np.float32)
    return {'mra_coords':coordinates(mi), 'mra_target':mra[mi,None].copy(),
            'sdf_coords':coordinates(si), 'sdf_target_mm':sdf[si,None].copy(),
            'sampling_report':{'mra':mr,'sdf':sr}}


class MRASDFDataset(Dataset):
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

    def __getitem__(self, index):
        if index not in self.cache:
            pair = load_pair(self.frame.iloc[index],self.config)
            if self.config['data']['cache_subjects'] > 0:
                self.cache[index] = pair
                while len(self.cache)>self.config['data']['cache_subjects']:
                    self.cache.popitem(last=False)
        else:
            pair = self.cache[index]
            self.cache.move_to_end(index)
        # Independent of worker scheduling/rank; fully reproducible at epoch boundaries.
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index]))
        result = sample_tasks(pair,self.config,rng)
        result['sampling_report'].update({'subject_index':int(index),'epoch':int(self.epoch)})
        result['condition'] = condition_vector(self.config,self.frame.iloc[index].scan_age,self.frame.iloc[index].sex)
        result['subject_index'] = int(index)
        return result


def collate_subjects(items):
    result = {k:torch.from_numpy(np.stack([item[k] for item in items])) for k in
              ('mra_coords','mra_target','sdf_coords','sdf_target_mm','condition')}
    result['subject_index'] = torch.tensor([item['subject_index'] for item in items],dtype=torch.long)
    result['sampling_report'] = [item['sampling_report'] for item in items]
    return result
