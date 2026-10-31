import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import SineLayer, MoESineLayer
from .config import field_names
from .graph_fields import FIELDS


def euler_rotation(angles):
    """Proper XYZ Euler rotation matrices (n, 3, 3) from angles (n, 3) in radians."""
    a,b,c = angles.unbind(-1)
    ca,sa,cb,sb,cc,sc = a.cos(),a.sin(),b.cos(),b.sin(),c.cos(),c.sin()
    rows = [cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa,
            sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa,
            -sb,cb*sa,cb*ca]
    return torch.stack(rows,dim=-1).reshape(-1,3,3)


def rigid_transform(coords, parameters, half_bbox):
    """Proper XYZ Euler rotations in physical mm; translation parameters are mm."""
    rotation = euler_rotation(parameters[:,:3])
    return (torch.einsum('nij,nj->ni',rotation,coords*half_bbox)+parameters[:,3:])/half_bbox


def linear_part(rotation, log_scale=None, shear=None):
    """A = R diag(exp(s)) H (mm -> mm), H unit upper-triangular with entries (h01, h02, h12)."""
    a = euler_rotation(rotation)
    if log_scale is not None:
        h = torch.eye(3, device=rotation.device, dtype=rotation.dtype).repeat(len(rotation), 1, 1)
        h[:, 0, 1], h[:, 0, 2], h[:, 1, 2] = shear[:, 0], shear[:, 1], shear[:, 2]
        a = a @ torch.diag_embed(log_scale.exp()) @ h
    return a


class SDFModel(nn.Module):
    """Single normalized-SDF head. Latent grids are indexed in physical (X,Y,Z) order."""
    def __init__(self, config, num_subjects):
        super().__init__()
        m = config['model']; self.model_config = m
        shape = m['latent_shape']
        self.latents = nn.Parameter(torch.randn(num_subjects,*shape)*m['latent_std'])
        # Subject -> template maps. legacy: one (N, 6) table [Euler rad, translation mm] (unchanged default).
        # normalized: separate tables (own learning rates): rotation (rad), translation in units of the bbox
        # half-lengths, and for affine log-scale + shear; optionally the population mean is fixed at identity.
        t = m.get('transform', {})
        self.transform_mode = t.get('mode', 'legacy'); self.transform_type = t.get('type', 'rigid')
        self.mean_identity = bool(t.get('mean_identity', False))
        if self.transform_mode == 'legacy':
            self.transforms = nn.Parameter(torch.zeros(num_subjects,6),requires_grad=m['rigid'])
        else:
            self.transforms = None
            self.transform_rotation = nn.Parameter(torch.zeros(num_subjects,3),requires_grad=m['rigid'])
            self.transform_translation = nn.Parameter(torch.zeros(num_subjects,3),requires_grad=m['rigid'])
            if self.transform_type == 'affine':
                self.transform_log_scale = nn.Parameter(torch.zeros(num_subjects,3),requires_grad=m['rigid'])
                self.transform_shear = nn.Parameter(torch.zeros(num_subjects,3),requires_grad=m['rigid'])
        self.latent_gate = 1.0  # 0 during an alignment stage: every subject decodes the shared template
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
        self.sdf_head = nn.Linear(head_features,1)
        bound = math.sqrt(6/m['hidden_size'])/m['omega_hidden']
        nn.init.uniform_(self.sdf_head.weight,-bound,bound)
        nn.init.zeros_(self.sdf_head.bias)
        # Optional Phase 4 heads (created only when enabled, so the baseline is unchanged).
        self.field_heads = nn.ModuleDict({name: nn.Linear(head_features, FIELDS[name]) for name in field_names(config)})
        # Alignment-stage head: distance to the large-vessel centreline (created only when alignment is enabled).
        if config.get('alignment', {}).get('enabled'):
            self.field_heads['align_distance'] = nn.Linear(head_features, 1)

    def subject_tensors(self):
        """Per-subject parameter tables (latents first), for optimizer groups and DDP synchronization checks."""
        names = ['latents'] + (['transforms'] if self.transform_mode == 'legacy' else
                               ['transform_rotation', 'transform_translation'] +
                               (['transform_log_scale', 'transform_shear'] if self.transform_type == 'affine' else []))
        return {name: getattr(self, name) for name in names}

    def transform_tables(self):
        """Effective normalized-mode tables (mean removed when mean_identity)."""
        tables = {k: v for k, v in self.subject_tensors().items() if k.startswith('transform_')}
        if self.mean_identity:
            tables = {k: v - v.mean(0, keepdim=True) for k, v in tables.items()}
        return tables

    def transform_coords(self, coords, subject_indices):
        """Subject-normalized coordinates -> template-normalized coordinates."""
        if self.transform_mode == 'legacy':
            return rigid_transform(coords, self.transforms[subject_indices], self.half_bbox)
        t = self.transform_tables()
        a = linear_part(t['transform_rotation'], t.get('transform_log_scale'), t.get('transform_shear'))[subject_indices]
        return torch.einsum('nij,nj->ni', a, coords * self.half_bbox) / self.half_bbox + t['transform_translation'][subject_indices]

    def maps_mm(self):
        """Per-subject maps in mm: template_mm = A @ (world - image_centre) + t_mm. Returns (A (N,3,3), t (N,3))."""
        with torch.no_grad():
            if self.transform_mode == 'legacy':
                return euler_rotation(self.transforms[:, :3]), self.transforms[:, 3:].clone()
            t = self.transform_tables()
            a = linear_part(t['transform_rotation'], t.get('transform_log_scale'), t.get('transform_shear'))
            return a, t['transform_translation'] * self.half_bbox

    def forward(self, coords, subject_indices, conditions, latent_override=None, apply_transform=True, balance_count=None,
                return_fields=False):
        """balance_count: MoE balancing uses only the first balance_count points (SDF samples)."""
        if coords.ndim!=2 or coords.shape[1]!=3 or conditions.shape!=(len(coords),2):
            raise ValueError('Expected Nx3 coordinates and Nx2 age/sex conditions')
        if subject_indices.shape!=(len(coords),):
            raise ValueError('Expected one subject index per point')
        if self.model_config['rigid'] and apply_transform and latent_override is None:
            coords = self.transform_coords(coords,subject_indices)
        if latent_override is not None:
            grids = self.modulator(latent_override)
            if grids.shape[0] != 1:
                raise ValueError('Inference latent_override must contain one latent grid')
            mod = F.grid_sample(grids,coords[:,[2,1,0]].reshape(1,1,1,-1,3),align_corners=True,padding_mode='border').reshape(grids.shape[1],-1).T
        else:
            subjects, inverse = torch.unique(subject_indices,sorted=True,return_inverse=True)
            grids = self.modulator(self.latents[subjects] * self.latent_gate)
            mod = coords.new_zeros((len(coords),self.latents.shape[1]))
            for j in range(len(subjects)):
                selected = torch.where(inverse==j)[0]
                grid = coords[selected][:,[2,1,0]].reshape(1,1,1,-1,3)
                values = F.grid_sample(grids[j:j+1],grid,align_corners=True,padding_mode='border').reshape(grids.shape[1],-1).T
                mod = mod.index_copy(0,selected,values)
        features = torch.cat([mod,conditions],dim=-1)
        for layer in self.backbone:
            if isinstance(layer,MoESineLayer):
                layer.balance_count = balance_count
        hidden,_ = self.backbone((coords,features))
        shared = torch.cat([hidden,features],dim=-1)
        # SDF deliberately stays linear, signed, and unbounded. Output shape (N, 1).
        output = self.sdf_head(shared)
        auxiliary = [layer.latest_loss for layer in self.backbone if isinstance(layer,MoESineLayer)]
        moe = sum(auxiliary)/len(auxiliary) if auxiliary else output.sum()*0.
        if return_fields:
            return output,moe,{name: head(shared) for name, head in self.field_heads.items()}
        return output,moe
