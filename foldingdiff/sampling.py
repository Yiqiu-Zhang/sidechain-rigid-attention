"""
Code for sampling from diffusion models
"""
import json
import os
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from huggingface_hub import snapshot_download

from foldingdiff import beta_schedules
from foldingdiff import utils
from foldingdiff import datasets as dsets
from foldingdiff import modelling
from write_preds_pdb import structure_build, geometry


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    coords:torch.Tensor,
    seq: torch.Tensor,
    t: torch.Tensor,
    chi_mask: torch.Tensor,
    acid_embedding: torch.Tensor,
    rigid_type: torch.Tensor,
    rigid_property: torch.Tensor,
    seq_lens: Sequence[int],
    t_index: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)

    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas) # alpha_beta_values
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"]) # alpha_beta_values["alphas"] ======= [alpha_t]

    # Select based on time
    t_unique = torch.unique(t) # t has to be single value?
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]
    
   # print("==================coords====================",coords.shape)
   # print("==================t====================",t.shape)
   # print("==================x====================",x.shape)
    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, l in enumerate(seq_lens):
        attn_mask[i, :l] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, coords, seq, t, chi_mask, acid_embedding, rigid_type, rigid_property)/ sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_x_0(
        model: nn.Module,
        x: torch.Tensor,
        coords: torch.Tensor,
        seq: torch.Tensor,
        t: torch.Tensor, # [Batch]
        chi_mask: torch.Tensor,
        acid_embedding: torch.Tensor,
        rigid_type: torch.Tensor,
        rigid_property: torch.Tensor,
        seq_lens: Sequence[int],
        t_index: torch.Tensor,
        betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)

    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)  # alpha_beta_values
    sqrt_alpha = torch.sqrt(alpha_beta_values["alphas"])
    sqrt_alphas_cumprod_prev = torch.sqrt(alpha_beta_values["alphas_cumprod_prev"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_alpha_t = sqrt_alpha[t_index]
    sqrt_alphas_cumprod_prev_t = sqrt_alphas_cumprod_prev[t_index]
    alphas_cumprod_t = alpha_beta_values["alphas_cumprod"][t_index]
    alphas_cumprod_prev_t = alpha_beta_values["alphas_cumprod_prev"][t_index]
    betas_t = betas[t_index]



    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, l in enumerate(seq_lens):
        attn_mask[i, :l] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
   # print("sqrt_alphas_cumprod_prev_t=",sqrt_alphas_cumprod_prev_t.shape)
   # print("sqrt_alphas_cumprod_prev_t=",sqrt_alphas_cumprod_prev_t)
   # print("betas_t=",betas_t.shape)
   # print("betas_t=",betas_t)
   # print("alphas_cumprod_t=",alphas_cumprod_t.shape)
   # print("alphas_cumprod_t=",alphas_cumprod_t)
   # print("sqrt_alpha_t=",sqrt_alpha_t.shape)
   # print("sqrt_alpha_t=",sqrt_alpha_t)
   # print(" alphas_cumprod_t=", alphas_cumprod_t.shape)
   # print(" alphas_cumprod_t=", alphas_cumprod_t)
   # print(" alphas_cumprod_prev_t=", alphas_cumprod_prev_t.shape)
   # print(" alphas_cumprod_prev_t=", alphas_cumprod_prev_t)
    
    sin_cos = model(x, coords, seq, t, acid_embedding, rigid_type, rigid_property,attn_mask)
    angles = torch.atan2(sin_cos[...,0],sin_cos[...,1])
    
    model_mean = sqrt_alphas_cumprod_prev_t * betas_t * angles / (1 - alphas_cumprod_t) \
                 + sqrt_alpha_t * (1 - alphas_cumprod_prev_t) * x / (1 - alphas_cumprod_t)


    #tempoutput = model(x, coords, seq, t, acid_embedding, rigid_type, rigid_property,attn_mask)
    #print("tempoutput=", angles.shape)
    #model_mean = sqrt_alphas_cumprod_prev_t * betas_t * model(x, coords, seq, t, acid_embedding, rigid_type, rigid_property,attn_mask) / (1 - alphas_cumprod_t) \
    #             + sqrt_alpha_t * (1 - alphas_cumprod_prev_t) * x / (1 - alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    coords:torch.Tensor,
    seq: torch.Tensor,
    chi_mask: torch.Tensor,
    acid_embedding: torch.Tensor,
    rigid_type: torch.Tensor,
    rigid_property: torch.Tensor,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps, disable=disable_pbar
    ):
        '''
        # Shape is (batch, seq_len, 4)
        img = p_sample(
            model=model,            
            x=img,
            coords=coords,
            seq=seq,
            t=torch.full((b,), i, device=device, dtype=torch.long), # time vector
            chi_mask=chi_mask,
            acid_embedding=acid_embedding,
            rigid_type=rigid_type,
            rigid_property=rigid_property,
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )
        '''

        img = p_sample_x_0(
            model=model,
            x=img,
            coords=coords,
            seq=seq,
            t=torch.full((b,), i, device=device, dtype=torch.long), # time vector
            chi_mask=chi_mask,
            acid_embedding=acid_embedding,
            rigid_type=rigid_type,
            rigid_property=rigid_property,
            seq_lens=lengths,
            t_index=i,
            betas=betas,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
    return torch.stack(imgs)

def rigid_apply_update(seq, bb_to_gb, delta_chi, current_local):
    return structure_build.torsion_to_frame(seq, bb_to_gb, delta_chi, current_local)

@torch.no_grad()
def p_sample_loop_score(
        model,
        coords:torch.Tensor,
        seq: torch.Tensor,
        acid_embedding: torch.Tensor,
        rigid_type: torch.Tensor,
        rigid_property: torch.Tensor,
        seq_lens: Sequence[int],
        corrupted_angles: torch.Tensor,

        sigma_max = torch.pi,
        sigma_min=0.01 * np.pi,
        steps=100,
):
    # [*, N_rigid, 4, 2]
    angles_sin_cos = torch.stack([torch.sin(corrupted_angles), torch.cos(corrupted_angles)], dim=-1)
    default_r = structure_build.get_default_r(seq, corrupted_angles)
    # [*, N_res] Rigid
    bb_to_gb = geometry.get_gb_trans(coords)
    # [*, N_rigid] Rigid
    rigids, current_local_frame = structure_build.torsion_to_frame(seq,
                                                             bb_to_gb,
                                                             angles_sin_cos,
                                                             default_r)

    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    # Create the attention mask
    pad_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, l in enumerate(seq_lens):
        pad_mask[i, :l] = 1.0

    imgs = []
    for sigma_idx, sigma in enumerate(sigma_schedule):
        score = model(seq,
                      rigids,
                      sigma,
                      acid_embedding,
                      rigid_type,
                      rigid_property,
                      pad_mask)

        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size= score.shape)
        perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

        rigids, current_local_frame = rigid_apply_update(seq, bb_to_gb, perturb, current_local_frame)
        torsion_angles = structure_build.rigids_to_torsion_angles(seq, rigids)
        imgs.append(torsion_angles.cpu())

    return torch.stack(imgs)

def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    structure, # a single data dictionary in the dataset
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 1,
    feature_key: str = "angles",
    disable_pbar: bool = False,
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    temp_c = structure["coords"]
    temp_c = torch.from_numpy(temp_c)
    temp_e = structure["acid_embedding"]
    temp_e = torch.from_numpy(temp_e)
    temp_s = structure["seq"]
    temp_s = torch.from_numpy(temp_s)
    temp_chi = structure["chi_mask"]
    temp_chi = torch.from_numpy(temp_chi)
    temp_rt = structure["rigid_type_onehot"]
    temp_rt = torch.from_numpy(temp_rt)
    temp_rp = structure["rigid_property"]
    temp_rp= torch.from_numpy(temp_rp)
    
    
    for this_lengths in lengths_chunkified:
        torch.cuda.empty_cache()
        batch = len(this_lengths)
        # batch is always one, should we set this batch size to N
        # so we can generate N sample at a single time ?????????
        # Sample noise and sample the lengths
        coords = temp_c.repeat(batch,1,1,1).cuda()
        acid_embedding = temp_e.repeat(batch,1,1).cuda()
        seq = temp_s.unsqueeze(1).repeat(batch,1,1).cuda().squeeze(-1)
        chi_mask = temp_chi.repeat(batch,1,1).cuda()
        rigid_type = temp_rt.repeat(batch,1,1,1).cuda()
        rigid_property = temp_rp.repeat(batch,1,1,1).cuda()

        # [b,n,f]
        noise = train_dset.sample_noise(
             torch.zeros((batch,
                          seq.shape[-1], # sequence length
                          model.n_inputs), # number of angles need to predict
                         dtype=torch.float64)
        )
        # Produces (timesteps, batch_size, seq_len, n_ft)
        this_lengths =  [seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1]]
        sampled = p_sample_loop_score(
            model=model,
            coords = coords,
            seq=seq,
            chi_mask=chi_mask,
            acid_embedding=acid_embedding,
            rigid_type=rigid_type,
            rigid_property=rigid_property,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular[feature_key],
            disable_pbar=disable_pbar
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval


def sample_simple(
    model_dir: str, n: int = 10, sweep_lengths: Tuple[int, int] = (50, 128)
) -> List[pd.DataFrame]:
    """
    Simple wrapper on sample to automatically load in the model and dummy dataset
    Primarily for gradio integration
    """
    if utils.is_huggingface_hub_id(model_dir):
        model_dir = snapshot_download(model_dir)
    assert os.path.isdir(model_dir)

    with open(os.path.join(model_dir, "training_args.json")) as source:
        training_args = json.load(source)

    model = modelling.BertForDiffusionBase.from_dir(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    dummy_dset = dsets.AnglesEmptyDataset.from_dir(model_dir)
    dummy_noised_dset = dsets.NoisedAnglesDataset(
        dset=dummy_dset,
        dset_key="coords" if training_args == "cart-cords" else "angles",
        timesteps=training_args["timesteps"],
        exhaustive_t=False,
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )

    sampled = sample(model, dummy_noised_dset, n=n, sweep_lengths=sweep_lengths, disable_pbar=True)
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=dummy_noised_dset.feature_names["angles"])
        for s in final_sampled
    ]
    return sampled_dfs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sample_simple("wukevin/foldingdiff_cath", n=1, sweep_lengths=(50, 55))
    for i, x in enumerate(s):
        print(x.shape)
