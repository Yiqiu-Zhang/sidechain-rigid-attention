from torch import nn
import torch
from typing import List

import math
import sys

from write_preds_pdb import structure_build
from write_preds_pdb.geometry import Rigid


""" IPA utils functions"""
def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

##############################################

class InputEmbedder(nn.Module):
    def __init__(
        self,
        nf_dim: int, # Node features dim
        pair_dim: int, # Pair features dim
        c_n: int,  # Node_embedding dim
        c_z: int, # Pair embedding dim
        relpos_k: int, # Window size of relative position
    ):
        self.nf_dim = nf_dim
        self.pair_dim = pair_dim

        self.c_z = c_z
        self.c_n = c_n

        self.linear_tf_z_i = nn.Linear(nf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(nf_dim, c_z)
        self.linear_tf_n = nn.Linear(nf_dim, c_n)

        # Relative_position encoding
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = nn.Linear(self.no_bins, c_z)


    def relpos(self,
        seq_len: int,
        batch_zize: int):

        rigid_res_idx = torch.arange(0, seq_len).unsqueeze(-1).repeat(1,5).reshape(-1)
        d = rigid_res_idx - rigid_res_idx[..., None]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(rigid_res_idx.dtype)
        l = len(d.shape)
        d = d.unsqueze(0).repeat(batch_zize,*(1,)*l) #  [B, N_rigid, N_rigid, C_pair]
        return self.linear_relpos(d) # [B, N_rigid, N_rigid, C_pair]

    def forward(self,
        noised_angles: torch.Tensor, #[batch,128,4]
        time_encoded: torch.Tensor,
        seq_onehot: torch.Tensor, #[batch,128,21]
        rigid_type: torch.Tensor, #[batch,128,5,20]
        rigid_property: torch.Tensor, #[batch,128,5,6]

        # 随后添加 pair embedding
        predict_angles: torch.Tensor, # [batch,128,4]
        distance: torch.Tensor, # [batch, N_rigid, N_rigid] distance 也要做分块处理比较好 dim = 16? []
        altered_direction: torch.Tensor, # [batch, N_rigid, N_rigid, 3]
        orientation: torch.Tensor# [batch, N_rigid, N_rigid] Rigid 要把这个东西变成quortnion
        ):
        
        assert rigid_property.shape[-1] == 6

        batch_size = seq_onehot.shape[0]
        seq_len = seq_onehot.shape[1]

        # [batch, N_rigid, c]
        flat_rigid_type = rigid_type.reshape(batch_size, -1, rigid_type.shape[-1])
        flat_rigid_proterty = rigid_property.reshape(batch_size, -1, rigid_property.shape[-1])
        expand_seq = seq_onehot.repeat(1,1,5).reshape(batch_size, -1, seq_onehot.shape[-1])

        # [batch, N_res, 8]
        sin_cos = torch.cat((torch.sin(noised_angles), torch.cos(noised_angles)), -1)
        expand_angle = sin_cos.repeat(1,1,5).reshape(batch_size, -1, sin_cos.shape[-1])

        # [batch, N_rigid, nf_dim]
        node_feature = torch.cat((flat_rigid_type, flat_rigid_proterty, expand_seq, expand_angle), dim=-1)


        nf_emb_i = self.linear_tf_z_i(node_feature)
        nf_emb_j = self.linear_tf_z_j(node_feature)

        # [*, N_rigid, N_rigid, c_z]
        relative_pos = self.relpos(seq_len, batch_size)

        # [*, N_rigid, N_rigid, c_z] = [*, N_rigid, 1, c_z] + [*, 1, N_rigid, c_z]
        pair_emb = nf_emb_i[..., None, :] + nf_emb_j[..., None, :, :]
        pair_emb = pair_emb + relative_pos

        # [*, N_rigid, c_n]
        node_emb = self.linear_tf_n(node_feature)

        # add time encode
        node_emb = node_emb + time_encoded
        pair_emb = pair_emb + time_encoded

        return node_emb, pair_emb

class InvariantPointAttention():

    def __init__(
            self,
            c_n: int, # node dim
            c_z: int, # edge dim
            c_hidden: int, # ipa dim = 12
            no_heads: int, # 8
            no_qk_points: int, # 4
            no_v_points: int, # 8
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        super(InvariantPointAttention, self).__init__()

        self.c_n = c_n # node dim
        self.c_z = c_z # edge dim
        self.c_hidden = c_hidden # ipa dim = 16
        self.no_heads = no_heads # 8
        self.no_qk_points = no_qk_points # 4
        self.no_v_points = no_v_points # 8
        self.inf = inf
        self.eps = eps

        # These linear layers differ from  Alphafold IPA module, Here we use standard nn.Linear initialization
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_n, hc)
        self.linear_kv = nn.Linear(self.c_n, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_n, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_n, hpkv)


        self.linear_b = nn.Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(concat_out_dim, self.c_n, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            r: Rigid,
            mask: torch.Tensor,

    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, c_n] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, c_n] single representation update
        """

        z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_rigid, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_rigid, no_heads, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_rigid, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_rigid, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1) # [*, N_rigid, H * P_q, 3]
        q_pts = r[..., None].apply(q_pts)  # rigid mut vec [*, N_rigid, 1] rigid * [*, N_rigid, H * P_q, 3]

        # [*, N_rigid, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_rigid, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_rigid, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts) # 这里v_pts也直接*了frame 在后面用到了

        # [*, N_rigid, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_rigid, N_rigid, H]
        b = self.linear_b(z[0])

        # [*, H, N_rigid, N_rigid]
        qT_k = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_rigid, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_rigid]
        )

        qT_k *= math.sqrt(1.0 / (3 * self.c_hidden)) # (3 * self.c_hidden) WL * c
        b = (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))) # [*,H, N_rigid, N_rigid]

        a = qT_k + b

        # [*, N_rigid, N_rigid, H, P_q, 3] = [*, N_rigid, 1, *] - [*, 1, N_rigid, *]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        pt_att = pt_att ** 2

        # [*, N_rigid, N_rigid, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1)) # calculate vector length
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_rigid, N_rigid, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5) # Sum over point
        # [*, N_rigid, N_rigid]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_rigid, N_rigid]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        # [*, H, N_rigid, N_rigid]
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_rigid, H, C_hidden] = [*, H, N_rigid, N_rigid] matmul [*,  H, N_rigid, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_rigid, H * C_hidden]
        o = flatten_final_dims(o, 2)


        # [*, H, 3, N_rigid, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None] # [*, H, 1, N_rigid, N_rigid, 1]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :] # [*,  H, 3, 1, N_rigid, P_v]
            ),
            dim=-2, # sum over j, the second N_rigid
        )

        # [*, N_rigid, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_rigid, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_rigid, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)


        # [*, N_rigid, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_rigid, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_rigid, c_n]  [*, N_rigid, H * C_hidden + H * P_v * 3 + H * P_v + H * C_z]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):

        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out


class TransitionLayer(nn.Module):
    def __init__(self, c):
        super(TransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = nn.Linear(self.c, self.c)
        self.linear_2 = nn.Linear(self.c, self.c)
        self.linear_3 = nn.Linear(self.c, self.c)

        self.relu = nn.ReLU()

        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        s = self.layer_norm

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed


class AngleResnet(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnet, self).__init__()

        super().__init__()
        self.c_hidden = c_hidden

        self.linear_1 = nn.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial

class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class RigidDiffusion(nn.Module):

    def __init__(self,config,
                 # InputEmbedder config
                 nf_dim: int = 51,
                 pair_dim: int = 0, # 暂时不知道 要等写完 pair embedding
                 c_n: int = 384,
                 c_z: int = 128,
                 relpos_k: int = 32,

                 num_blocks:int = 3,
                 # IPA config
                 c_hidden: int = 12,  # ipa dim = 12
                 no_heads: int = 8,  # 8
                 no_qk_points: int =4,  # 4
                 no_v_points: int =8,  # 8
                 ):
        super(RigidDiffusion, self).__init__()
        self._ipa_config = config
        
        self.input_embedder = InputEmbedder(nf_dim, pair_dim, c_n, c_z, relpos_k)
        self.trunk = nn.ModuleDict()

        for b in range(num_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(c_n, c_z,
                                                             c_hidden,
                                                             no_heads,
                                                             no_qk_points,
                                                             no_v_points)

            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_n)
            '''
            self.trunk[f'skip_embed_{b}'] = nn.Linear(
                self._model_conf.node_embed_size,
                self._ipa_conf.c_skip,
                init="final"
            )
            '''
            self.trunk[f'node_transition_{b}'] = TransitionLayer(c_n)
            self.trunk[f'angle_resnet_{b}'] = AngleResnet(c_n)

            if b < config.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=config.c_n,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.rigid_transformer_stack = nn.ModuleList([InvariantPointAttention() for _ in range(n_layers)])
        self.angles_predictor = AnglesPredictor(d_model * 5, d_angles)
        self.noise_predictor = NoisePredictor(d_model * 5, d_angles)
        self.transition = TransitionLayer()
        self.angle_resnet = AngleResnet()

    def forward(self,
                side_chain_angles,
                backbone_coords,
                seq_idx,
                time_encoded,
                extended_attention_mask,
                x_seq_esm,
                x_rigid_type,
                x_rigid_proterty,
        ):
        # TEMP
        predict_angles = torch.zeros(1)

        rigids = structure_build.torsion_to_frame(seq_idx,
                                                 backbone_coords,
                                                 side_chain_angles)  # add attention #frame

        # [*, N_rigid, N_rigid, c]
        frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids, seq_idx)

        # [*, N_rigid, c_n], [*, N_rigid, N_rigid, c_z]
        init_node_emb, pair_emb = self.input_embedder(side_chain_angles,
                                                         time_encoded,
                                                         x_seq_esm,
                                                         x_rigid_type,
                                                         x_rigid_proterty,
                                                         predict_angles,
                                                         distance,
                                                         altered_direction,
                                                         orientation)
        # 初始设定 包括recycling用法
        node_emb = torch.clone(init_node_emb)

        for b in range(self._ipa_config.num_blocks):
            if b > 0:
                rigids = structure_build.torsion_to_frame(seq_idx,
                                                          backbone_coords,
                                                          updated_chi_angles)  # add attention #frame

                # [*, N_rigid, N_rigid, c]
                frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids,
                                                                                                          seq_idx)
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_emb,
                pair_emb,
                rigids,
                frame_pair_mask) # 这里还不确定要传入哪个mask

            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)




            # [*, N_rigid] Rigid


            node_emb, pair_emb = InvariantPointAttention(node_emb_initial,node_emb, pair_emb, rigids, frame_pair_mask)

            # [*, N_res, c_n * 5] Reshape N_rigid into N_res
            node_emb = node_emb.reshape(node_emb.shape[0], -1, node_emb.shape[-1] * 5)
            node_emb = self.transition(node_emb)
            updated_chi_angles = self.trunk[f'angle_resnet_{b}']()

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # 或许正确的操作应该是 预测noise 用加噪音后的角度-noise 继续， 而不是直接预测角度 然后最后预测noise
        noise = self.noise_predictor(node_emb)


        return noise

