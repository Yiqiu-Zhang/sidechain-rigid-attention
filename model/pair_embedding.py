from torch import nn
import torch
from functools import partial
from typing import Optional

from triangular_attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
from triangular_multiplicative_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
from pair_transition import PairTransition
from primitives import LayerNorm, Linear, Attention

class PairStackBlock(nn.Module):
    def __init__(
            self,
            c_z: int,
            c_hidden_tri_att: int,
            c_hidden_tri_mul: int,
            no_heads: int,
            pair_transition_n: int,
            inf: float,
    ):
        super(PairStackBlock, self).__init__()

        self.c_z = c_z
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.inf = inf

        #self.dropout_row = DropoutRowwise(self.dropout_rate)
        #self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_z,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_z,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_z,
            self.c_hidden_tri_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_z,
            self.c_hidden_tri_mul,
        )

        self.pair_transition = PairTransition(
            self.c_z,
            self.pair_transition_n,
        )

    def forward(self,
                z: torch.Tensor,
                mask: torch.Tensor,
                _mask_trans: bool = True,
                ):

        z_mask = mask

        z = z + self.tri_att_start(z, mask=z_mask)


        z = z + self.tri_att_end(z, mask=z_mask)


        tmu_update = self.tri_mul_out(z, mask=z_mask)

        z = z + tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(z, mask=z_mask)

        z = z + tmu_update

        del tmu_update

        z = z + self.pair_transition(z,mask=z_mask if _mask_trans else None)

        return z


class PairStack(nn.Module):
    """
    Implements Algorithm 16.
    """
    def __init__(
        self,
        c_z,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        inf=1e9,
    ):
        """
        Args:
            c_z:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
        """
        super(PairStack, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = PairStackBlock(c_z,
                                   c_hidden_tri_att,
                                   c_hidden_tri_mul,
                                   no_heads,
                                   pair_transition_n,
                                   inf)

            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_z)

    def forward(
        self,
        pair_emb: torch.tensor,
        mask: torch.tensor,
        _mask_trans: bool = True, # 这个东西是干什么的
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, c_z] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, c_z] template embedding update
        """
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = pair_emb.shape[-4]
            mask = mask.expand(*expand_idx)

        for pair_block in  self.blocks:
            pair_emb = pair_block(pair_emb, mask = mask, _mask_trans = _mask_trans)

        pair_emb = self.layer_norm(pair_emb)

        return pair_emb

class PairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        pair_dim,
        c_z,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(PairEmbedder, self).__init__()

        self.pair_stack = PairStack(c_z,
                                    c_hidden_tri_att,
                                    c_hidden_tri_mul,
                                    no_blocks,
                                    no_heads,
                                    pair_transition_n)

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = nn.Linear(pair_dim, c_z, init="relu")

    def forward(
        self,
        pair_feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
            [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        pair_emb = self.linear(pair_feature)
        pair_emb = self.pair_stack(pair_emb, mask)

        return x
