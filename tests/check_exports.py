import json
from pathlib import Path
import numpy as np
import nibabel as nib
import sys
root=Path(sys.argv[1]);result={}
for folder,prefix in [('atlas_age20_M_spacing1_SMOKE_ONLY','atlas'),('subject_native_SMOKE_ONLY','subject')]:
    directory=root/folder
    names=('MRA_atlas','VesselSDF_atlas','VesselMask_atlas') if prefix=='atlas' else ('Subject_MRA_Reconstruction','Subject_VesselSDF','Subject_VesselMask')
    images=[nib.load(directory/f'{name}.nii.gz') for name in names]
    arrays=[image.get_fdata() for image in images]
    assert all(np.isfinite(x).all() for x in arrays)
    assert len({image.shape for image in images})==1
    assert all(np.allclose(image.affine,images[0].affine) for image in images)
    assert np.array_equal(arrays[2],arrays[1]<0)
    if prefix=='atlas':
        assert images[0].shape==(240,240,120)
        np.testing.assert_allclose(images[0].affine,np.array([[1,0,0,-120],[0,1,0,-120],[0,0,1,-60],[0,0,0,1]]))
    else:
        original=nib.load('/leonardo/home/userexternal/xzhang01/data/v1/data/MRA/IXI002-Guys-0828-MRA.nii.gz')
        assert original.shape==images[0].shape
        np.testing.assert_allclose(original.affine,images[0].affine)
    result[folder]={'shape':images[0].shape,'affine':images[0].affine.tolist(),
                    'mra_range':[float(arrays[0].min()),float(arrays[0].max())],
                    'sdf_range_mm':[float(arrays[1].min()),float(arrays[1].max())],
                    'vessel_voxels':int(arrays[2].sum()),'zero_level_mismatch_voxels':0}
(root/'export_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
