from rigid_diffusion_score import RigidDiffusion
import torch
from write_preds_pdb import  structure_build, geometry
batch = 2
n_rigid = 25
n_res = 5
side_chain_angles = torch.randn((batch,n_res,4))
backbone_coords = torch.randn((batch,n_res,4,3))
seq_idx = torch.randint(0,20,(batch,n_res))
seq_esm = torch.randn((batch,n_res,320))
sigma = torch.rand((batch,1))* torch.pi
#timesteps = torch.randint(0,1000,(batch,1))
rigid_type = torch.randn((batch, n_res, 5, 20))
rigid_property = torch.randn((batch,n_res,5,6))
pad_mask = torch.randint(0,1, (batch,n_res))
ture_angles = torch.randn((batch,n_res,4))
diffusion_mask = torch.randint(0,1, (batch,n_res,1)) == 1

angles_sin_cos = torch.stack([torch.sin(side_chain_angles), torch.cos(side_chain_angles)], dim=-1)
default_r = structure_build.get_default_r(seq_idx, side_chain_angles)
# [*, N_res] Rigid
bb_to_gb = geometry.get_gb_trans(backbone_coords)
# [*, N_rigid] Rigid
rigids, current_local_frame = structure_build.torsion_to_frame(seq_idx,
                                                               bb_to_gb,
                                                               angles_sin_cos,
                                                               default_r)
model = RigidDiffusion()
run = model.forward(#side_chain_angles,
                    #ture_angles,
                    #backbone_coords,
                    seq_idx,
                    #diffusion_mask,
                    rigids,
                    sigma,
                    seq_esm,
                    rigid_type,
                    rigid_property,
                    pad_mask,)

