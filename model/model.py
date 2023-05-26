from torch import nn
import torch
from typing import List, Tuple

import math

from write_preds_pdb import structure_build
from write_preds_pdb.geometry import Rigid

from pair_embedding import PairEmbedder
from utils import permute_final_dims, flatten_final_dims, ipa_point_weights_init_, rbf, quaternions
from primitives import LayerNorm, Linear

##############################################
class AngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(AngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = nn.Linear(self.c_in, self.c_out)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.c_out, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x

class InputEmbedder(nn.Module):
    def __init__(
        self,
        nf_dim: int, # Node features dim
        c_n: int,  # Node_embedding dim
        relpos_k: int, # Window size of relative position

        # Pair Embedder parameter
        pair_dim: int, # Pair features dim
        c_z: int, # Pair embedding dim
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_blocks: int,
        no_heads: int,
        pair_transition_n: int,
    ):
        super(InputEmbedder, self).__init__()
        self.nf_dim = nf_dim

        self.c_z = c_z
        self.c_n = c_n

        # self.angle_embedder = AngleEmbedder()
        self.pair_embedder = PairEmbedder(
                                        pair_dim,
                                        c_z,
                                        c_hidden_tri_att,
                                        c_hidden_tri_mul,
                                        no_blocks,
                                        no_heads,
                                        pair_transition_n)

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
        time_encoded: torch.Tensor, # 记得要把time encode 改一下改成 适合 pair 和 适合 node的
        seq_onehot: torch.Tensor, #[batch,128,21]
        rigid_type: torch.Tensor, #[batch,128,5,20]
        rigid_property: torch.Tensor, #[batch,128,5,6]

        # predict_angles: torch.Tensor,  [batch,128,4] 需要这个吗？
        distance: torch.Tensor, # [batch, N_rigid, N_rigid] distance 也要做分块处理比较好 （做了_rbf）
        altered_direction: torch.Tensor, # [batch, N_rigid, N_rigid, 3]
        orientation: torch.Tensor# [batch, N_rigid, N_rigid] Rigid 要把这个东西变成 quaternion
        ):
        
        assert rigid_property.shape[-1] == 6

        batch_size = seq_onehot.shape[0]
        seq_len = seq_onehot.shape[1]

        # [batch, N_rigid, c]
        flat_rigid_type = rigid_type.reshape(batch_size, -1, rigid_type.shape[-1])
        flat_rigid_proterty = rigid_property.reshape(batch_size, -1, rigid_property.shape[-1])
        expand_seq = seq_onehot.repeat(1,1,5).reshape(batch_size, -1, seq_onehot.shape[-1])

        # [batch, N_rigid, 8]
        sin_cos = torch.cat((torch.sin(noised_angles), torch.cos(noised_angles)), -1)
        expand_angle = sin_cos.repeat(1,1,5).reshape(batch_size, -1, sin_cos.shape[-1])

        """ 目前我们直接把 Angle 拼上去 然后做一个linear，后期可以尝试把 angle 单独拿出来 
        做一个linear，relu，linear 然后再拼回去试一下"""
        # [batch, N_rigid, nf_dim]
        node_feature = torch.cat((flat_rigid_type, flat_rigid_proterty, expand_seq, expand_angle), dim=-1)

        # [*, N_rigid, c_n]
        node_emb = self.linear_tf_n(node_feature)

        ################ Pair_feature ####################

        # [batch, N_rigid, N_rigid, C_x] C_x = 23?
        distance_rbf = rbf(distance)
        orientation_quaternions = quaternions(orientation)
        pair_feature = torch.cat((distance_rbf, altered_direction, orientation_quaternions),dim=-1)

        pair_emb = self.pair_embedder(pair_feature)

        nf_emb_i = self.linear_tf_z_i(node_feature)
        nf_emb_j = self.linear_tf_z_j(node_feature)

        # [*, N_rigid, N_rigid, c_z]
        relative_pos = self.relpos(seq_len, batch_size)

        # [*, N_rigid, N_rigid, c_z] = [*, N_rigid, 1, c_z] + [*, 1, N_rigid, c_z]
        pair_emb = pair_emb + nf_emb_i[..., None, :] + nf_emb_j[..., None, :, :]

        pair_emb = pair_emb + relative_pos

        # add time encode
        node_emb = node_emb + time_encoded
        pair_emb = pair_emb + time_encoded

        return node_emb, pair_emb

class InvariantPointAttention(nn.Module):

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
                [*, N_rigid, c_n] single representation
            z:
                [*, N_rigid, N_rigid, C_z] pair representation
            r:
                [*, N_rigid] transformation object
            mask:
                [*, N_rigid] mask
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
        square_mask = self.inf * (square_mask - 1) # 这里靠 mask 逼近 -inf 之后再用 softmax 让 attention score 变 0

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

class TransitionLayer(nn.Module):
    """ We only get one transitionlayer in our model, so no module needed."""
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

        s = self.layer_norm(s)

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
        self.initial_embed = nn.Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = nn.Linear(hidden_size, edge_embed_out, init="final")
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
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, no_rigids, epsilon):
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

        self.c_in = c_in * no_rigids
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_in, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles * 2)

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



        # [*, N_res, c_n * 5]
        # 这里把不同 rigid的信息拼起来在这里
        s_initial = s_initial.reshape(s_initial.shape[0], -1, s_initial.shape[-1] * 5)
        s = s.reshape(s.shape[0], -1, s.shape[-1] * 5)

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

class RigidDiffusion(nn.Module):

    def __init__(self,
                 num_blocks: int = 3, # 整个网络的循环次数

                 # InputEmbedder config
                 nf_dim: int = 51,
                 c_n: int = 384, # Node channel dimension after InputEmbedding
                 relpos_k: int = 32, # relative position neighbour range

                 # Pair Embedder parameter
                 pair_dim: int = 23, # rbf + direction_vector + qu
                 c_z: int = 128, # Pair channel dimension after InputEmbedding
                 c_hidden_tri_att: int =16, # Keep ori
                 c_hidden_tri_mul: int =64, # Keep ori
                 pairemb_no_blocks: int = 2, # Keep ori
                 mha_no_heads: int = 4, # Keep ori
                 pair_transition_n: int = 2, # Keep ori

                 # IPA config
                 c_hidden: int = 12,  # IPA hidden channel dimension
                 ipa_no_heads: int = 8,  # Number of attention head
                 no_qk_points: int =4,  # Number of qurry/key (3D vector) point head
                 no_v_points: int =8,  # Number of value (3D vector) point head

                 # Change of channel dimension from c_n to cn*5 cause of rigid
                 c_res: int = 384*5,

                 # AngleResnet
                 c_resnet: int = 128, # AngleResnet hidden channel dimension
                 no_resnet_blocks: int = 2, # Resnet block number
                 no_angles: int = 4, # predict chi 1-4 4 angles
                 no_rigids: int = 5, # number of rigids to concate togather
                 epsilon: int = 1e-8,

                 ):

        super(RigidDiffusion, self).__init__()

        self.num_blocks = num_blocks
        
        self.input_embedder = InputEmbedder(nf_dim, c_n, relpos_k,
                                            pair_dim, c_z,
                                            c_hidden_tri_att, c_hidden_tri_mul,
                                            pairemb_no_blocks, mha_no_heads, pair_transition_n,
        )
        self.trunk = nn.ModuleDict()

        for b in range(num_blocks):
            # [*, N_res, c_n * 5]
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(c_n,
                                                             c_z,
                                                             c_hidden,
                                                             ipa_no_heads,
                                                             no_qk_points,
                                                             no_v_points)

            self.trunk[f'ipa_ln_{b}'] = LayerNorm(c_n)
            '''
            self.trunk[f'skip_embed_{b}'] = nn.Linear(
                self._model_conf.node_embed_size,
                self._ipa_conf.c_skip,
                init="final"
            )
            '''
            self.trunk[f'node_transition_{b}'] = TransitionLayer(c_n)
            self.trunk[f'angle_resnet_{b}'] = AngleResnet(c_n,
                                                          c_resnet,
                                                          no_resnet_blocks,
                                                          no_angles,
                                                          no_rigids,
                                                          epsilon)

            if b < num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(c_n, c_z, c_z)

    def forward(self,
                side_chain_angles,
                backbone_coords,
                seq_idx,
                time_encoded,
                extended_attention_mask,
                seq_onehot,
                rigid_type,
                rigid_property,
        ):

        rigids = structure_build.torsion_to_frame(seq_idx,
                                                 backbone_coords,
                                                 side_chain_angles)  # add attention #frame

        # flat_mask [*, N_rigid]， others [*, N_rigid, N_rigid, c]
        pair_mask, flat_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids, seq_idx)

        # [*, N_rigid, c_n], [*, N_rigid, N_rigid, c_z]
        init_node_emb, pair_emb = self.input_embedder(side_chain_angles,
                                                      time_encoded,
                                                      seq_onehot,
                                                      rigid_type,
                                                      rigid_property,
                                                      distance,
                                                      altered_direction,
                                                      orientation)
        # 初始设定 包括recycling用法
        node_emb = torch.clone(init_node_emb)

        for b in range(self.num_blocks):

            if b > 0:
                rigids = structure_build.torsion_to_frame(seq_idx,
                                                          backbone_coords,
                                                          updated_chi_angles)  # add attention #frame

                # [*, N_rigid, N_rigid, c]
                flat_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids,
                                                                                                          seq_idx)

            ipa_embed = self.trunk[f'ipa_{b}'](node_emb, pair_emb, rigids, flat_mask)

            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            updated_chi_angles = self.trunk[f'angle_resnet_{b}'](node_embed, init_node_emb)

            if b < self.num_blocks-1:
                pair_emb = self.trunk[f'edge_transition_{b}'](node_embed, pair_emb)
                pair_emb *= pair_mask[..., None]

            # [*, N_res, c_n * 5]
            # Reshape N_rigid into N_res 这里其实一直没有好好写， 这五个rigid的表示直接被拼起来就用来预测角度了，这里是不是应该换一种方法？
            # 直接放到 IPA 里面怎么样
            # node_emb = node_emb.reshape(node_emb.shape[0], -1, node_emb.shape[-1] * 5)

        # 或许正确的操作应该是 预测noise 用加噪音后的角度-noise 继续， 而不是直接预测角度 然后最后预测noise
        noise = self.noise_predictor(node_emb)

        return noise

