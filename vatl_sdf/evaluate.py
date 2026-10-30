"""Aligned-grid SDF evaluation. The reference vessel mask is reference SDF < 0 (= label 1).

Connectivity is always reported relative to the subject's reference mask: the reference
data are themselves fragmented, so a prediction that reproduces the reference's pieces is
correct, and one connected component is not the target.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

STRUCTURE_26 = np.ones((3, 3, 3), bool)


def dice(a,b):
    total=a.sum()+b.sum()
    return float(2*(a&b).sum()/total) if total else 1.


def components(mask, affine):
    labels,count=ndi.label(mask,STRUCTURE_26)
    sizes=np.bincount(labels.ravel())[1:]
    return {'count_26':int(count),'largest_sizes_voxels':sorted(sizes.tolist(),reverse=True)[:20],
            'vessel_voxels':int(mask.sum()),'vessel_volume_mm3':float(mask.sum()*abs(np.linalg.det(affine[:3,:3])))}


def connectivity(mask, min_voxels=10):
    labels,count=ndi.label(mask,STRUCTURE_26)
    sizes=np.bincount(labels.ravel())[1:]
    return {'components':int(count),'components_ge_min':int((sizes>=min_voxels).sum()),
            'largest_fraction':float(sizes.max()/sizes.sum()) if count else 0.}


def reference_relative_connectivity(pred_mask, reference_mask, min_voxels=10):
    """26-connected component counts and largest-component fraction, prediction vs reference.

    component_ratio > 1: more pieces than the reference (fragmentation beyond the data);
    largest_fraction_difference < 0: the main tree holds less of the vessel volume than in
    the reference. *_ge_min ignores components smaller than min_voxels.
    """
    p, r = connectivity(pred_mask, min_voxels), connectivity(reference_mask, min_voxels)
    ratio = lambda a, b: float(a/b) if b else None
    return {'pred_components':p['components'],'reference_components':r['components'],
            'component_ratio':ratio(p['components'],r['components']),
            'component_excess':p['components']-r['components'],
            'pred_components_ge_min':p['components_ge_min'],'reference_components_ge_min':r['components_ge_min'],
            'component_ratio_ge_min':ratio(p['components_ge_min'],r['components_ge_min']),
            'pred_largest_fraction':p['largest_fraction'],'reference_largest_fraction':r['largest_fraction'],
            'largest_fraction_difference':p['largest_fraction']-r['largest_fraction'],'min_voxels':min_voxels}


def evaluate(sdf, sdf_reference, affine, tau, exported_mask=None):
    if sdf.shape!=sdf_reference.shape:
        raise ValueError('Reference and predicted fields must have identical shapes')
    if not (np.isfinite(sdf).all() and np.isfinite(sdf_reference).all()):
        raise ValueError('Evaluation requires finite fields')
    band=np.abs(sdf_reference)<=tau
    vessel=sdf<0;reference=sdf_reference<0
    result={'sdf':{'mae_mm':float(np.abs(sdf-sdf_reference).mean()),
                   'truncation_region_mae_mm':float(np.abs(sdf[band]-sdf_reference[band]).mean()) if band.any() else None,
                   'clipped_mae_mm':float(np.abs(np.clip(sdf,-tau,tau)-np.clip(sdf_reference,-tau,tau)).mean()),
                   'zero_level_dice_vs_reference_sdf':dice(vessel,reference),
                   'sign_agreement':float((vessel==reference).mean()),
                   'components':components(vessel,affine),
                   'reference_relative_connectivity':reference_relative_connectivity(vessel,reference)}}
    if exported_mask is not None:
        if exported_mask.shape!=sdf.shape:raise ValueError('Exported mask shape mismatch')
        result['sdf']['export_zero_level_mismatch_voxels']=int((vessel!=exported_mask).sum())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sdf',required=True);p.add_argument('--sdf-reference',required=True)
    p.add_argument('--exported-mask');p.add_argument('--tau',type=float,default=5.)
    p.add_argument('--output');a=p.parse_args()
    main_image=nib.load(a.sdf)
    def load(path):
        image=nib.load(path)
        if image.shape!=main_image.shape or not np.allclose(image.affine,main_image.affine,atol=1e-3):
            raise ValueError('All volumes must already share the identical grid/affine')
        return image.get_fdata()
    result=evaluate(load(a.sdf),load(a.sdf_reference),main_image.affine,a.tau,
                    load(a.exported_mask)>0 if a.exported_mask else None)
    text=json.dumps(result,indent=2);print(text)
    if a.output:
        with Path(a.output).open('x') as stream:stream.write(text)

if __name__=='__main__':main()
