import argparse
import json
from pathlib import Path
import torch
import pandas as pd
import nibabel as nib
from .config import condition_vector
from .model import MRASDFModel
from .inference import generate_atlas,native_grid,predict_field,export_fields


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--age',type=float);p.add_argument('--sex',choices=['M','F'])
    p.add_argument('--chunk-size',type=int);p.add_argument('--spacing',nargs=3,type=float);p.add_argument('--average-fields',action='store_true')
    p.add_argument('--subject-id',help='native-grid reconstruction of a checkpoint training subject')
    a=p.parse_args();checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['config'];frame=pd.DataFrame(checkpoint['metadata'])
    if a.spacing:config['atlas']['spacing_mm']=a.spacing
    if a.chunk_size:config['atlas']['chunk_size']=a.chunk_size
    model=MRASDFModel(config,len(frame)).to(a.device);model.load_state_dict(checkpoint['model']);model.eval()
    if a.subject_id:
        matches=frame.index[frame.subject_id==a.subject_id]
        if len(matches)!=1:raise ValueError('Subject ID must belong to the checkpoint training set')
        index=int(matches[0]);row=frame.iloc[index];image=nib.load(row.MRA)
        coords,shape,affine=native_grid(image,config)
        output=predict_field(model,coords,condition_vector(config,row.scan_age,row.sex),config,subject_index=index)
        directory=export_fields(output,shape,affine,a.output,config['sdf']['truncation_mm'],subject=True)
        (directory/'provenance.json').write_text(json.dumps({'subject_id':row.subject_id,'MRA':row.MRA,'SDF':row.SDF,
            'scan_age':float(row.scan_age),'sex':row.sex,'checkpoint':str(Path(a.checkpoint).resolve())},indent=2))
    else:
        directory=generate_atlas(model,frame,config,a.output,a.age,a.sex,a.average_fields)
        (directory/'checkpoint.txt').write_text(str(Path(a.checkpoint).resolve()))
    print(directory)

if __name__=='__main__':main()
