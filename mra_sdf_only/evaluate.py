"""Aligned-grid evaluation. Optional vessel reference is evaluation-only."""
import argparse
import json
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity


def dice(a,b):
    total=a.sum()+b.sum()
    return float(2*(a&b).sum()/total) if total else 1.


def components(mask, affine):
    labels,count=ndi.label(mask)
    sizes=np.bincount(labels.ravel())[1:]
    return {'count':int(count),'largest_sizes_voxels':sorted(sizes.tolist(),reverse=True)[:20],
            'vessel_voxels':int(mask.sum()),'vessel_volume_mm3':float(mask.sum()*abs(np.linalg.det(affine[:3,:3])))}


def evaluate(mra, sdf, mra_reference, sdf_reference, affine, tau, vessel_reference=None, exported_mask=None):
    if any(a.shape!=mra.shape for a in (sdf,mra_reference,sdf_reference)):
        raise ValueError('Reference and predicted fields must have identical shapes')
    if not all(np.isfinite(a).all() for a in (mra,sdf,mra_reference,sdf_reference)):
        raise ValueError('Evaluation requires finite fields')
    mse=float(np.mean((mra-mra_reference)**2));band=np.abs(sdf_reference)<=tau
    vessel=sdf<0;reference=sdf_reference<0
    result={'mra':{'mae':float(np.abs(mra-mra_reference).mean()),
                   'psnr_range_1_db':float(-10*np.log10(mse)) if mse else None,
                   'ssim_range_1':float(structural_similarity(mra,mra_reference,data_range=1.,win_size=min(7,min(mra.shape)//2*2-1)))},
            'sdf':{'mae_mm':float(np.abs(sdf-sdf_reference).mean()),
                   'truncation_region_mae_mm':float(np.abs(sdf[band]-sdf_reference[band]).mean()) if band.any() else None,
                   'clipped_mae_mm':float(np.abs(np.clip(sdf,-tau,tau)-np.clip(sdf_reference,-tau,tau)).mean()),
                   'zero_level_dice_vs_reference_sdf':dice(vessel,reference),
                   'sign_agreement':float((vessel==reference).mean()),
                   'components':components(vessel,affine)}}
    if vessel_reference is not None:
        if vessel_reference.shape!=sdf.shape:raise ValueError('Vessel reference shape mismatch')
        result['sdf']['vessel_reference_dice']=dice(vessel,vessel_reference)
    if exported_mask is not None:
        if exported_mask.shape!=sdf.shape:raise ValueError('Exported mask shape mismatch')
        result['sdf']['export_zero_level_mismatch_voxels']=int((vessel!=exported_mask).sum())
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ('mra','sdf','mra-reference','sdf-reference'):p.add_argument('--'+key,required=True)
    p.add_argument('--vessel-reference',help='evaluation-only vessel mask or integer labels')
    p.add_argument('--vessel-label',type=int,default=1)
    p.add_argument('--config',help='apply the training intensity normalization to the MRA reference');p.add_argument('--exported-mask');p.add_argument('--tau',type=float,default=5.)
    p.add_argument('--output');a=p.parse_args()
    main_image=nib.load(a.mra)
    def load(path):
        image=nib.load(path)
        if image.shape!=main_image.shape or not np.allclose(image.affine,main_image.affine,atol=1e-3):
            raise ValueError('All volumes must already share the identical grid/affine')
        return image.get_fdata()
    # Normalize the reference using the exact target normalization and saved mra_scale
    # before evaluation when the source images were not already in [0,1].
    reference=load(a.mra_reference)
    if a.config:
        from .config import load_config
        from .dataset import normalize_mra
        reference,_=normalize_mra(reference,load_config(a.config))
    result=evaluate(load(a.mra),load(a.sdf),reference,load(a.sdf_reference),main_image.affine,a.tau,
                    load(a.vessel_reference)==a.vessel_label if a.vessel_reference else None,
                    load(a.exported_mask)>0 if a.exported_mask else None)
    text=json.dumps(result,indent=2);print(text)
    if a.output:
        from pathlib import Path
        with Path(a.output).open('x') as stream:stream.write(text)

if __name__=='__main__':main()
