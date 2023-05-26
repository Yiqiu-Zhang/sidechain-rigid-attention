"""
structure_transformer
"""
import os, sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from transformers.activations import get_activation
from typing import *
sys.path.append("../write_preds_pdb")
import structure_build
seaborn.set_context(context="talk") 


# define embedding of ridig and esm-seq
class Ridig_Embeddings(nn.Module):
    def __init__(self, d_model, d_esm_seq, n_rigid_type, n_rigid_property):
        super(Ridig_Embeddings, self).__init__()
       # self.embed_rigid_type = nn.Embedding(n_rigid_type, d_model)
       # self.embed_rigid_property = nn.Embedding(n_rigid_property, d_model)
        self.embed_rigid_type = nn.Linear(n_rigid_type, d_model, dtype=torch.float64)
        self.embed_rigid_property = nn.Linear(n_rigid_property, d_model, dtype=torch.float64)
        self.seq2d = nn.Linear(d_esm_seq, d_model) 
        self.mlp2d = nn.Sequential(
                  nn.Linear(d_model+d_model+d_model, d_model//2, dtype=torch.float64),
                  nn.ReLU(),
                  nn.Linear(d_model//2, d_model, dtype=torch.float64)
        )
        #self.embed_rigid_idx =  nn.Linear(5, d_model, dtype=torch.float64)

    def forward(
        self, 
        x_seq_esm: torch.Tensor, 
        x_rigid_type: torch.Tensor,
        x_rigid_proterty: torch.Tensor,
    ) -> torch.Tensor: 
        x_embed_rigid_type = self.embed_rigid_type(x_rigid_type) #[batch, L, 5, 128]
        x_embed_rigid_proterty = self.embed_rigid_property(x_rigid_proterty) #[batch, L, 5, 128]
        x_seq = self.seq2d(x_seq_esm) #[batch, L, 512]
        x_seq = x_seq.unsqueeze(-2)
        input = torch.cat([x_embed_rigid_type, x_embed_rigid_proterty,  x_seq.repeat(1, 1, 5, 1)], dim=-1)  #[batch, L, 5, 128*3]
        input_embed = self.mlp2d(input) #[batch, L, 5, 128]
        # 这里可以把 mlp2d 撤掉试试看





        # pair embedding

        p_ij = relattive_pose()
        seq_ij = x_seq + x_seq.transpose()  # z_ij = a_i + b_j
        d_ij = distance()


        return input_embed
    
# define sinusoidal positional encoding 1
class acid_SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len =5000):
        super(acid_SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # dropout = 0.2 or 0.1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # (1000) -> (1000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term) #
        pe = pe.unsqueeze(0) # (1000,512)->(1,1000,512) 
        self.register_buffer('pe', pe) # trainable parameters of the model
    
    def forward(self, x):
        y = x.sum(dim=-2)
        temp_pe = self.pe[:, :y.size(1)] #[batch,128,384]
        temp_pe = temp_pe.unsqueeze(-2).repeat(1, 1, 5, 1) #[batch,128,5,384]
        x = x + temp_pe
        return self.dropout(x)

class SinusoidalPositionalEncoding(nn.Module):
    # Implement PE function
    def __init__(self, d_model, dropout, max_len =1000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # dropout = 0.2 or 0.1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # (1000) -> (1000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term) #
        pe = pe.unsqueeze(0) # (1000,512)->(1,1000,512) 
        self.register_buffer('pe', pe) # trainable parameters of the model
    
    def forward(self, x):
         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
         return self.dropout(x)
 
# define relative positional encoding 2
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len =1000):
        super(RelativePositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.rpe = nn.Parameter(torch.randn(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        seq_len = x.size(1)

        pos1 = torch.arange(seq_len).unsqueeze(1).expand(-1, seq_len)
        pos2 = torch.arange(seq_len).unsqueeze(0).expand(seq_len, -1)
        pos_diff = pos2 - pos1 #relative position
        
        rp_embedding = torch.zeros(seq_len, seq_len, x.size(-1), device=x.device)
        for k in range(x.size(-1)):
            rp_embedding[:, :, k] = torch.sin(pos_diff / 10000 ** (2 * k / x.size(-1)))
        
        rp_embedding = torch.matmul(rp_embedding, self.rpe) 
        
        x = x + self.pe.weight.unsqueeze(0).unsqueeze(0) + rp_embedding.unsqueeze(0)

        return self.dropout(x)
    
# define convolutional relative positional encoding 2        
class ConvRelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len =1000):
        super(ConvRelativePositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False) #relative positional encoding based on conv
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        seq_len = x.size(1)

        pos1 = torch.arange(seq_len).unsqueeze(1).expand(-1, seq_len)
        pos2 = torch.arange(seq_len).unsqueeze(0).expand(seq_len, -1)
        pos_diff = pos2 - pos1 #relative position
        
        rp_embedding = torch.zeros(seq_len, seq_len, x.size(-1), device=x.device)
        for k in range(x.size(-1)):
            rp_embedding[:, :, k] = torch.sin(pos_diff / 10000 ** (2 * k / x.size(-1)))

        rp_embedding = rp_embedding.transpose(1, 2).contiguous()
        rp_embedding = self.conv(rp_embedding) # relative positional encoding using conv
        
        x = x + self.pe.weight.unsqueeze(0).unsqueeze(0) + rp_embedding.unsqueeze(0)

        return self.dropout(x)       

def calculate_rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 16
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, 1, 1, -1]).to('cuda')
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1).to('cuda')
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

# define the multi ridge attention
class Rigid_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Rigid_MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.head_dim_3d = 3
        # has no bias 这里linear in,out dim 为什么是一样的呀 那这个linear没有维度转换做的完全意义
        self.query = nn.Linear(d_model, d_model, bias=False, dtype=torch.float64)
        self.key = nn.Linear(d_model, d_model, bias=False, dtype=torch.float64)
        self.value = nn.Linear(d_model, d_model, bias=False, dtype=torch.float64)
        self.value_3d = nn.Linear(d_model, 3, bias=False, dtype=torch.float64)
        self.fc = nn.Linear(d_model+3, d_model, dtype=torch.float64)
        self.mlp =  nn.Sequential(
                  nn.Linear(16, 8, dtype=torch.float64),
                  nn.ReLU(),
                  nn.Linear(8, 1, dtype=torch.float64)
        )

        
    def forward(self, x_rigid, altered_direction, orientation, distance, attention_mask= None, frame_pair_mask=None, seq_correlation_matrix=None, ):
        bsz = x_rigid.size(0)
        q = self.query(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, rigid_len, head_dim] [batch, 8, 128*5, 96]
        k = self.key(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, 8, 128*5, 96]
        v = self.value(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, 8, 128*5, 96]
        v_3d = self.value_3d(x_rigid) #[batch, 128*5, 3] 
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float)) #[batch, n_heads, rigid_len, rigid_len] [batch,8,128*5,128*5]
       # print('===========================distance==================',distance)
        rbf = calculate_rbf(distance)
        #print("==================rbf1=================",rbf.shape)

        rbf = rbf.unsqueeze(1).repeat(1, 8, 1,1,1).double()
        #print("==================rbf2=================",rbf.shape)
        
        rbf = self.mlp(rbf)

        #print("==================rbf3=================",rbf.shape)
        dis = rbf.view(rbf.shape[0],rbf.shape[1],rbf.shape[2],rbf.shape[3])
        #print("==================rbf4=================",dis.shape)
        scores = scores + dis
        #print("==================rbf4=================",scores.shape)
        attention_mask = attention_mask.unsqueeze(-3).to('cuda')
        attention_mask = attention_mask.repeat(1, 8, 1, 1).to('cuda')
        if attention_mask is not None:
           # Mask invalid positions
           scores = scores.masked_fill_(attention_mask == 0, -1e9) 
        
        attn_weights = F.softmax(scores, dim=-1) # [batch,8,128*5,128*5] n_heads = 8
        attn_v = torch.matmul(attn_weights, v)   # [batch,8,128*5,128*5] # 应该是 [batch,8,128*5, 96] ？
        attn_v = attn_v.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)  # [batch,128*5,384] # 最后这一步没看懂
        socres_merged = torch.sum(scores, dim=-3).squeeze(dim=-3) #
        attn_weights_merged = F.softmax(socres_merged, dim=-1) # [batch,128*5,128*5]
        
        v3d_map = torch.einsum('bijmn,bjn->bijn', orientation, v_3d)  # [batch,128*5,128*5,3]
        v3d_direction = torch.cross(altered_direction, v3d_map, dim=-1)  # [batch,128*5,128*5,3]

        # 这个地方的理解有偏差 绝对不应该出现attn_weights_merged 这个东西
        attn_frame = torch.mul(attn_weights_merged.unsqueeze(-1), v3d_direction)  # [batch,128*5,128*5,3] = [batch,128*5,128*5,1] dot [batch,128*5,128*5,3]
        attn_frame = torch.mean(attn_frame,dim=-2)   # [batch,128*5,128*5,3] -> [batch,128*5,3]
        attn_output = torch.cat((attn_v, attn_frame), dim=-1) 
        attn_output = self.fc(attn_output) # [batch,128*5,384+3]->[batch,128*5,3]

        return attn_output

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

class InvariantPointAttention(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension = 16
            no_heads:
                Number of attention heads = 8
            no_qk_points:
                Number of query/key points to generate = 4
            no_v_points:
                Number of value points to generate = 8
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s # node dim
        self.c_z = c_z # edge dim
        self.c_hidden = c_hidden # ipa dim = 16
        self.no_heads = no_heads # 8
        self.no_qk_points = no_qk_points # 4
        self.no_v_points = no_v_points # 8
        self.inf = inf
        self.eps = eps

        # These linear layers differ from  Alphafold IPA module, Here we use standard nn.Linear initialization
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)


        self.linear_b = nn.Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            inplace_safe: bool = False,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if (_offload_inference and inplace_safe):
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, no_heads, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1) # [*, N_res, H * P_q, 3]
        q_pts = r[..., None].apply(q_pts)  # rigid mut vec [*, N_res, 1] rigid * [*, N_res, H * P_q, 3]

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            assert (sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        qT_k = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )

        qT_k *= math.sqrt(1.0 / (3 * self.c_hidden)) # (3 * self.c_hidden) WL * c
        b = (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        a = qT_k + b

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if (inplace_safe):
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if (inplace_safe):
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)


        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, dtype=torch.float64)
        self.fc2 = nn.Linear(d_ff, d_model, dtype=torch.float64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = Rigid_MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, dtype=torch.float64)
        self.norm2 = nn.LayerNorm(d_model, dtype=torch.float64)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
   
    def forward(self, x_rigid, altered_direction,  orientation, rigid_mask, distance, frame_pair_mask=None, seq_correlation_matrix=None):
        # Self-attention
        residual = x_rigid
        x_rigid = self.norm1(x_rigid)
        
        x_rigid = self.self_attn(x_rigid, altered_direction, orientation, rigid_mask, distance)
        x_rigid = residual +self.dropout1(x_rigid)

        # Feed forward
        residual = x_rigid
        x_rigid = self.norm2(x_rigid)
        x_rigid = self.ff(x_rigid)
        x_rigid= self.dropout2(x_rigid)
        x_rigid = residual + x_rigid

        return x_rigid

class AnglesPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int = 4,
        activation: Union[str, nn.Module] = "gelu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model//5, dtype=torch.float64)
        #拉平了去做，后面可以是5个一组
        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm_angle = nn.LayerNorm(d_model//5, eps=eps, dtype=torch.float64)
        # 这个有点没看懂 为啥要做 layerNorm
        self.dense2 = nn.Linear(d_model//5, d_out, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm_angle(x)
        x = self.dense2(x)
        return x
    
class NoisePredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model//5, dtype=torch.float64)
        self.dense2 = nn.Linear(d_model, d_model//5, dtype=torch.float64)
        self.dense3 =nn.Sequential(
                  nn.ReLU(),
                  nn.Linear(d_model//5, d_model//5, dtype=torch.float64),
                  nn.ReLU(),
                  nn.Linear(d_model//5, d_model//5, dtype=torch.float64)
        )
        self.dense4 = nn.Sequential(
                  nn.ReLU(),
                  nn.Linear(d_model//5, d_model//5, dtype=torch.float64),
                  nn.ReLU(),
                  nn.Linear(d_model//5, d_model//5, dtype=torch.float64)

        )
        self.dense5 = nn.Sequential(
                  nn.ReLU(),
                  nn.Linear(d_model//5, d_out, dtype=torch.float64)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x1)+self.dense2(x2)
        x = x + self.dense3(x)
        x = x + self.dense4(x)
        x = self.dense5(x)
        return x

# 填坑：
# 1. seq_pair信息没放到attetion中作为最终输出的一部分；
# 2. distance未作为尺度加到attenion权重；初步设想，是引入某个投影函数，对distance做数值变化，让其作为attention的约束。
# rigid的距离太远or太近都不应该发挥太大作用。【类似neighbor思想？】
# 3. bias未加入attention权重计算；
# 4. frame的attention并没有使用多头，而是合并了权重，记得加入多头机制；
# 5. 位置编码需要重新更改；
# 6. 预测角度，MLP的输入是[batch,l,r*5,d] 是否改为[batch,l,r,5,d]
# [batch,seq_len,rigid,d] 

#====================================================最初的版本=================================

class Ridge_Transformer(nn.Module):
    def __init__(self,
        d_model: int = 384,
        d_esm_seq: int = 320,
        n_rigid_type: int = 20,
        n_rigid_property: int = 6,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024, #hidden layer dim
        d_angles:int = 4,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):  
        super(Ridge_Transformer, self).__init__()
        self.embedding = Ridig_Embeddings(d_model, d_esm_seq, n_rigid_type, n_rigid_property) 
        self.pos_encoding = acid_SinusoidalPositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.predict_angles = AnglesPredictor(d_model*5, d_angles)
        self.predict_noise = NoisePredictor(d_model*5,d_angles)
        self.norm = nn.LayerNorm(d_model, dtype=torch.float64)

    def forward(self,
        side_chain_angles: torch.Tensor, #[batch,128,4]
        backbone_coords: torch.Tensor, #[batch,128,4,3]
        aatype_idx: torch.Tensor,#[batch,128,1]
        time_encoded: torch.Tensor, 
        rigid_mask: torch.Tensor,
        x_seq_esm: torch.Tensor,  #[batch,128,1024]
        x_rigid_type: torch.Tensor, #[batch,128,5,19] x_rigid_type[-1]=one hot
        x_rigid_proterty: torch.Tensor, #[batch,128,5,6]
    ):
     #   print("========rigid transformer start=========")
        x_rigid = self.embedding(x_seq_esm, x_rigid_type, x_rigid_proterty) # [batch,128,5,384]
     #   print("======== x_rigid  embedding=========")

        x_rigid = self.pos_encoding(x_rigid) #rigid finish # [batch,128,5,384]

        edge_embedding =
        
        x_rigid = x_rigid.view(x_rigid.size(0), -1, x_rigid.size(-1)) # [batch,128*5,384]
        
        x_rigid_11  = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2]//5, 5, x_rigid.shape[-1])
        x_rigid_22 = x_rigid.reshape(x_rigid_11.shape[0], x_rigid_11.shape[-3], -1)  
        x_rigid_init = x_rigid_22
        x_rigid = x_rigid+time_encoded
        for layer in self.layers:
            rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords, side_chain_angles) # add attention #frame 
            frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue, aatype_idx)
            x_rigid = layer(x_rigid, altered_direction,  orientation, frame_pair_mask, distance) # [batch,128, 5, 384]
            x_rigid = self.norm(x_rigid)
            x_rigid_1  = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2]//5, 5, x_rigid.shape[-1])
            x_rigid_2 = x_rigid.reshape(x_rigid_1.shape[0], x_rigid_1.shape[-3], -1)     
            side_chain_angles = self.predict_angles(x_rigid_2) #[batch,128,4]
        noise = self.predict_noise(x_rigid_2, x_rigid_init)
        return noise


class Rigid_Transformer(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 d_esm_seq: int = 320,
                 n_rigid_type: int = 20,
                 n_rigid_property: int = 6,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 d_ff: int = 1024,  # hidden layer dim
                 d_angles: int = 4,
                 max_seq_len: int = 5000,
                 dropout: float = 0.1,
                 relpos_k: int = 32,
                 pair_embeding_dim: int = 384,
                 ):
        super(Rigid_Transformer, self).__init__()
        self.c_z = pair_embeding_dim

        self.embedding = Ridig_Embeddings(d_model, d_esm_seq, n_rigid_type, n_rigid_property)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.predict_angles = AnglesPredictor(d_model * 5, d_angles)
        self.predict_noise = NoisePredictor(d_model * 5, d_angles)
        self.norm = nn.LayerNorm(d_model, dtype=torch.float64)

        self.edge_embedding = nn.Linear(edge_in, pair_embeding_dim, bias=True)

        # Relative_position encoding
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = nn.Linear(self.no_bins, pair_embeding_dim)

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
                side_chain_angles: torch.Tensor,  # [batch,128,4]
                backbone_coords: torch.Tensor,  # [batch,128,4,3]
                aatype_idx: torch.Tensor,  # [batch,128,1]
                time_encoded: torch.Tensor,
                rigid_mask: torch.Tensor,
                x_seq_esm: torch.Tensor,  # [batch,128,1024]
                x_rigid_type: torch.Tensor,  # [batch,128,5,19] x_rigid_type[-1]=one hot
                x_rigid_proterty: torch.Tensor,  # [batch,128,5,6]
                max_seq_len: int,
                ):
        #   print("========rigid transformer start=========")
        x_rigid = self.embedding(x_seq_esm, x_rigid_type, x_rigid_proterty)  # [batch,128,5,384]
        #   print("======== x_rigid  embedding=========")

        x_rigid = x_rigid.view(x_rigid.size(0), -1, x_rigid.size(-1))  # [batch,128*5,384]

        x_rigid_11 = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2] // 5, 5, x_rigid.shape[-1])
        x_rigid_22 = x_rigid.reshape(x_rigid_11.shape[0], x_rigid_11.shape[-3], -1)
        x_rigid_init = x_rigid_22
        x_rigid = x_rigid + time_encoded


        ###########
        # pair features calculation
        ###########

        #calculate frame from torsion angle
        rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords, side_chain_angles)
        frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue,
                                                                                                  aatype_idx)
        rbf = self._rbf(distance)
        q = self._quaternions(orientation)


        # [B, N_rigid, N_rigid, C_pair]
        relative_pos = self.relpos(max_seq_len)

        edge = torch.cat((relative_pos, rbf, altered_direction, q), -1)
        edge = self.edge_embedding(edge)
        edge = self.norm_edges(edge)

        # 这里是不是应该给 edge_embedding 加上 time eocoding


        for layer in self.layers:
            rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords,
                                                                side_chain_angles)  # add attention #frame
            frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue,
                                                                                                      aatype_idx)
            x_rigid = layer(x_rigid, altered_direction, orientation, frame_pair_mask, distance)  # [batch,128, 5, 384]
            x_rigid = self.norm(x_rigid)
            x_rigid_1 = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2] // 5, 5, x_rigid.shape[-1])
            x_rigid_2 = x_rigid.reshape(x_rigid_1.shape[0], x_rigid_1.shape[-3], -1)
            side_chain_angles = self.predict_angles(x_rigid_2)  # [batch,128,4]
        noise = self.predict_noise(x_rigid_2, x_rigid_init)
        return noise

class RigidDiffusion(nn.Module):

    def __init__(self):

        super(RigidDiffusion, self).__init__()

        self.input_embedder = InputEmbedder()
        self.rigid_transformer_stack = nn.ModuleList([RigidTransformer() for _ in range(n_layers)])
        self.angles_predictor = AnglesPredictor(d_model * 5, d_angles)
        self.noise_predictor = NoisePredictor(d_model * 5, d_angles)

    def forward(self,batch):

        h, z = self.input_embedder()

        for rigid_transformer in self.layers:

            h, z = rigid_transformer()

            side_chain_angles = self.angles_predictor(h)

        noise = self.noise_predictor(h)

        return noise

