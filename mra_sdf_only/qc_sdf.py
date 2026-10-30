"""Optional OFFLINE SDF sign check against a vessel-only reference.

This reference is evaluation-only and is never passed into dataset/model/loss.
An integer-label file can be used solely to extract the specified vessel label.
"""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from .evaluate import dice


def main():
    p=argparse.ArgumentParser();p.add_argument('--sdf',required=True);p.add_argument('--vessel-reference',required=True)
    p.add_argument('--vessel-label',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
    image=nib.load(a.sdf);reference=nib.load(a.vessel_reference)
    if image.shape!=reference.shape or not np.allclose(image.affine,reference.affine,atol=1e-3):raise ValueError('Reference grid differs')
    sdf=image.get_fdata();mask=reference.get_fdata()==a.vessel_label
    if not np.isfinite(sdf).all():raise ValueError('Nonfinite distances')
    edge=ndi.binary_dilation(mask)^ndi.binary_erosion(mask)
    result={'sdf':a.sdf,'vessel_reference':a.vessel_reference,'vessel_label':a.vessel_label,
            'negative_inside_dice':dice(sdf<0,mask),'sign_agreement':float(((sdf<0)==mask).mean()),
            'boundary_sign_agreement':float(((sdf[edge]<0)==mask[edge]).mean()) if edge.any() else None,
            'spacing_mm':nib.affines.voxel_sizes(image.affine).tolist(),'range_mm':[float(sdf.min()),float(sdf.max())],
            'reference_role':'offline vessel-only QC; not a training input'}
    with Path(a.output).open('x') as stream:stream.write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
