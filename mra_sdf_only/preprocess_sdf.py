"""Optional OFFLINE vessel-only EDT construction. Never invoked by training."""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


def signed_distance(vessel,affine):
    spacing=nib.affines.voxel_sizes(affine)
    directions=affine[:3,:3]/spacing
    if not np.allclose(directions.T@directions,np.eye(3),atol=1e-3):
        raise ValueError('Voxel EDT requires orthogonal physical axes; resample a sheared grid first')
    if not vessel.any() or vessel.all():raise ValueError('Vessel mask must have both inside and outside voxels')
    return (ndi.distance_transform_edt(~vessel,sampling=spacing)-ndi.distance_transform_edt(vessel,sampling=spacing)).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--vessel-mask',required=True);p.add_argument('--vessel-label',type=int,default=1);p.add_argument('--output',required=True)
    a=p.parse_args();destination=Path(a.output)
    if destination.exists():raise FileExistsError(destination)
    image=nib.load(a.vessel_mask)
    if image.header.get_xyzt_units()[0] not in ('mm','unknown'):raise ValueError('Expected affine millimetres')
    vessel=image.get_fdata()==a.vessel_label
    sdf=signed_distance(vessel,image.affine)
    out=nib.Nifti1Image(sdf,image.affine);out.header.set_xyzt_units('mm');nib.save(out,destination)

if __name__=='__main__':main()
