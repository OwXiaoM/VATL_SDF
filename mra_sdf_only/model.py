import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import SineLayer, MoESineLayer


def rigid_transform(coords, parameters, half_bbox):
    """Proper XYZ Euler rotations in physical mm; translation parameters are mm."""
    a,b,c = parameters[:,:3].unbind(-1)
    ca,sa,cb,sb,cc,sc = a.cos(),a.sin(),b.cos(),b.sin(),c.cos(),c.sin()
    rows = [cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa,
            sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa,
            -sb,cb*sa,cb*ca]
    rotation = torch.stack(rows,dim=-1).reshape(-1,3,3)
    return (torch.einsum('nij,nj->ni',rotation,coords*half_bbox)+parameters[:,3:])/half_bbox


class MRASDFModel(nn.Module):
    """Two named heads. Latent grids are indexed in physical (X,Y,Z) order."""
    def __init__(self, config, num_subjects):
        super().__init__()
        m = config['model']; self.model_config = m
        shape = m['latent_shape']
        self.latents = nn.Parameter(torch.randn(num_subjects,*shape)*m['latent_std'])
        self.transforms = nn.Parameter(torch.zeros(num_subjects,6),requires_grad=m['rigid'])
        self.register_buffer('half_bbox',torch.tensor(config['data']['world_bbox_mm'],dtype=torch.float32)/2.)
        k = m['convolution_kernel']
        if k and (k < 1 or k%2==0):
            raise ValueError('Latent convolution kernel must be zero or positive odd')
        self.modulator = nn.Conv3d(shape[0],shape[0],k,padding=k//2) if k else nn.Identity()
        latent_features = shape[0]+2
        layers = [SineLayer(3,latent_features if 0 in m['modulated_layers'] else 0,m['hidden_size'],is_first=True,omega=m['omega_first'])]
        for index in range(m['hidden_layers']):
            lat_size = latent_features if index+1 in m['modulated_layers'] else 0
            if m['use_moe'] and 0 < index < m['hidden_layers']-1:
                layer = MoESineLayer(m['hidden_size'],lat_size,m['hidden_size'],m['num_experts'],m['moe_k'],m['omega_hidden'])
            else:
                layer = SineLayer(m['hidden_size'],lat_size,m['hidden_size'],omega=m['omega_hidden'])
            layers.append(layer)
        self.backbone = nn.Sequential(*layers)
        head_features = m['hidden_size']+latent_features
        self.mra_head = nn.Linear(head_features,1)
        self.sdf_head = nn.Linear(head_features,1)
        for head in (self.mra_head,self.sdf_head):
            nn.init.uniform_(head.weight,-math.sqrt(6/m['hidden_size'])/m['omega_hidden'],math.sqrt(6/m['hidden_size'])/m['omega_hidden'])
            nn.init.zeros_(head.bias)

    def forward(self, coords, subject_indices, conditions, latent_override=None, apply_transform=True):
        if coords.ndim!=2 or coords.shape[1]!=3 or conditions.shape!=(len(coords),2):
            raise ValueError('Expected Nx3 coordinates and Nx2 age/sex conditions')
        if subject_indices.shape!=(len(coords),):
            raise ValueError('Expected one subject index per point')
        if self.model_config['rigid'] and apply_transform and latent_override is None:
            coords = rigid_transform(coords,self.transforms[subject_indices],self.half_bbox)
        if latent_override is not None:
            grids = self.modulator(latent_override)
            if grids.shape[0] != 1:
                raise ValueError('Inference latent_override must contain one latent grid')
            mod = F.grid_sample(grids,coords[:,[2,1,0]].reshape(1,1,1,-1,3),align_corners=True,padding_mode='border').reshape(grids.shape[1],-1).T
        else:
            subjects, inverse = torch.unique(subject_indices,sorted=True,return_inverse=True)
            grids = self.modulator(self.latents[subjects])
            mod = coords.new_zeros((len(coords),self.latents.shape[1]))
            for j in range(len(subjects)):
                selected = torch.where(inverse==j)[0]
                grid = coords[selected][:,[2,1,0]].reshape(1,1,1,-1,3)
                values = F.grid_sample(grids[j:j+1],grid,align_corners=True,padding_mode='border').reshape(grids.shape[1],-1).T
                mod = mod.index_copy(0,selected,values)
        features = torch.cat([mod,conditions],dim=-1)
        hidden,_ = self.backbone((coords,features))
        shared = torch.cat([hidden,features],dim=-1)
        mra = self.mra_head(shared)
        if self.model_config['mra_activation']=='sigmoid':
            mra = mra.sigmoid()
        # SDF deliberately stays linear, signed, and unbounded.
        output = torch.cat([mra,self.sdf_head(shared)],dim=-1)
        auxiliary = [layer.latest_loss for layer in self.backbone if isinstance(layer,MoESineLayer)]
        moe = sum(auxiliary)/len(auxiliary) if auxiliary else output.sum()*0.
        return output,moe
