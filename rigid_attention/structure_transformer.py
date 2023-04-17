"""
structure_transformer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from transformers.activations import get_activation
from typing import *
import structure_build
seaborn.set_context(context="talk") 


# define embedding of ridig and esm-seq
class Ridig_Embeddings(nn.Module):
    def __init__(self, d_model, d_esm_seq, n_rigid_type, n_rigid_property):
        super(Ridig_Embeddings, self).__init__()
        self.embed_rigid_type = nn.Embedding(n_rigid_type, d_model)
        self.embed_rigid_property = nn.Embedding(n_rigid_property, d_model)
        self.seq2d = nn.Linear(d_esm_seq, d_model) 
        self.mlp2d = nn.Sequential(
                  nn.Linear(d_model+d_model+d_model, d_model//2),
                  nn.ReLU(),
                  nn.Linear(d_model//2, d_model)
        )
        self.d_model = d_model

    def forward(
        self, 
        x_seq_esm: torch.Tensor, 
        x_rigid_type: torch.Tensor,
        x_rigid_proterty: torch.Tensor,
    ) -> torch.Tensor: 
        x_embed_rigid_type = self.embed_rigid_type(x_rigid_type)* math.sqrt(self.d_model) #[batch, L, 5, 128]
        x_embed_rigid_proterty = self.embed_rigid_property(x_rigid_proterty)* math.sqrt(self.d_model) #[batch, L, 5, 128]
        x_seq = self.seq2d(x_seq_esm) #[batch, L, 512]
        
        x_seq = x_seq.unsqueeze(-2)
        input = torch.cat([x_embed_rigid_type, x_embed_rigid_proterty,  x_seq.repeat(1, 1, 4, 1)], dim=-1)  #[batch, L, 5, 128*3]
        
        input_embed = self.mlp2d(input) #[batch, L, 4, 128]
        
        return input_embed
    
# define sinusoidal positional encoding 1
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
        #self.conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False) #relative positional encoding based on conv
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


# define the multi ridge attention
class Rigid_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Rigid_MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.head_dim_3d = 3
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.value_3d = nn.Linear(d_model, 3)
        self.fc = nn.Linear(d_model+3, d_model)

    def forward(self, x_rigid, altered_direction, orientation, attention_mask=None, frame_pair_mask=None, seq_correlation_matrix=None, distance =None):
        bsz = x_rigid.size(0)
        q = self.query(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, n_heads, rigid_len, head_dim] [batch, 8, 128*5, 96]
        k = self.key(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, 8, 128*5, 96]
        v = self.value(x_rigid).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [batch, 8, 128*5, 96]
        #v_3d = self.value_3d(x_rigid).view(bsz, -1, self.n_heads, 3).transpose(1, 2) # (batch, n_heads, rigid_len, 3) [batch, 8, 128*5, 3]
        v_3d = self.value_3d(x_rigid) #[batch, 128*5, 3]
        
        #seq_correlation_matrix # (batch, n_heads, seq_len, seq_len)
    
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float)) #[batch, n_heads, rigid_len, rigid_len] [batch,8,128*5,128*5]
        if attention_mask is not None:
           # Mask invalid positions
           scores = scores.masked_fill_(attention_mask == 0, -1e9) 
        
        attn_weights = F.softmax(scores, dim=-1) # [batch,8,128*5,128*5] n_heads = 8
        attn_v = torch.matmul(attn_weights, v)   # [batch,8,128*5,128*5]
        attn_v = attn_v.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)  # [batch,128*5,384]
        
        socres_merged = torch.sum(scores, dim=-3).squeeze(dim=-3)
        attn_weights_merged = F.softmax(socres_merged, dim=-1) # [batch,128*5,128*5]
        
        
        #v_3d = v_3d.unsqueeze(1).unsqueeze(-1)
        

        v3d_map = torch.einsum('bijmn,bjn->bijn', orientation, v_3d)  # [batch,128*5,128*5,3]
        v3d_direction = torch.cross(altered_direction, v3d_map,dim=-1)  # [batch,128*5,128*5,3]
        
       
        attn_frame = torch.mul(attn_weights_merged.unsqueeze(-1), v3d_direction)  # [batch,128*5,128*5,3] = [batch,128*5,128*5,1] dot [batch,128*5,128*5,3]
        attn_frame = torch.mean(attn_frame,dim=-2)   # [batch,128*5,128*5,3] -> [batch,128*5,3]
         
       # attn_pari = torch.matmul(attn_weights_merged, seq_correlation_matrix)   
       # attn_pari = attn_pari.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)  # (bsz, rigid_len, d_model)   
       # attn_frame = attn_frame.transpose(1, 2).contiguous().view(bsz, -1, 3)  # (bsz, rigid_len, d_model)
        
        attn_output = torch.cat((attn_v, attn_frame), dim=-1) 
        
        attn_output = self.fc(attn_output) # [batch,128*5,384+3]->[batch,128*5,3]

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = Rigid_MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
   
    def forward(self, x_rigid, altered_direction,  orientation, rigid_mask, frame_pair_mask=None, seq_correlation_matrix=None, distance =None):
        # Self-attention
        residual = x_rigid
        x_rigid = self.norm1(x_rigid)
        
        x_rigid = self.self_attn(x_rigid, altered_direction, orientation, rigid_mask)
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
        self.dense1 = nn.Linear(d_model, d_model)
        #拉平了去做，后面可以是5个一组
        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


# 填坑：
# 1. seq_pair信息没放到attetion中作为最终输出的一部分；
# 2. distance未作为尺度加到attenion权重；初步设想，是引入某个投影函数，对distance做数值变化，让其作为attention的约束。rigid的距离太远or太近都不应该发挥太大作用。【类似neighbor思想？】
# 3. bias未加入attention权重计算；
# 4. frame的attention并没有使用多头，而是合并了权重，记得加入多头机制；
# 5. 位置编码需要重新更改；
# 6. 预测角度，MLP的输入是[batch,l,r*5,d] 是否改为[batch,l,r,5,d]
# [batch,seq_len,rigid,d] 

class Ridge_Transformer(nn.Module):
    def __init__(self,
        d_model: int = 384,
        d_esm_seq: int = 1024,
        n_rigid_type: int = 5,
        n_rigid_property: int = 20,
        n_layers: int = 5,
        n_heads: int = 8,
        d_ff: int = 1024, #hidden layer dim
        d_angles:int = 4,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):  
        print("===========================ridig transformer paramters===========================")
        print("===============d_model===============", d_model)
        print("===============d_esm_seq===============", d_esm_seq)
        print("===============n_rigid_type===============", n_rigid_type)
        print("===============n_rigid_property===============", n_rigid_property)   
        print("===============n_layers===============", n_layers)
        print("===============n_heads===============", n_heads)
        print("===============d_ff===============", d_ff)
        print("===============d_angles===============", d_angles)
        print("===============max_seq_len===============", max_seq_len)
        print("===============dropout===============", dropout)
        print("===========================ridig transformer paramters===========================")
        super(Ridge_Transformer, self).__init__()
        self.embedding = Ridig_Embeddings(d_model, d_esm_seq, n_rigid_type, n_rigid_property) 
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout, max_seq_len)
        self.linear = nn.Linear(d_esm_seq, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.predict_angles = AnglesPredictor(d_model, d_angles)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
        side_chain_angles: torch.Tensor, #[batch,128,4]
        backbone_coords: torch.Tensor, #[batch,128,4,3]
        aatype_idx: torch.Tensor,#[batch,128,4,3]
        time: torch.Tensor, 
        rigid_mask: torch.Tensor,
        x_seq_esm: torch.Tensor,  #[batch,128,1024]
        x_rigid_type: torch.Tensor, #[batch,128,5,19] x_rigid_type[-1]=one hot
        x_rigid_proterty: torch.Tensor, #[batch,128,5,6]
    ):
        x_rigid = self.embedding(x_seq_esm, x_rigid_type, x_rigid_proterty) # [batch,128,5,384]
        x_rigid = x_rigid.view(x_rigid.size(0), -1, x_rigid.size(-1)) # [batch,128*5,384]
        x_rigid = self.pos_encoding(x_rigid) #rigid finish # [batch,128*5,384]
        

        #calculate correlation matrix based on 
        x_seq_esm = self.linear(x_seq_esm)  # [batch,128,384]
        
        #==============================================================================#
        #seq_correlation_matrix= calculate_correlation_matrix(x_seq_esm) #  引入pair信息，填坑
        #==============================================================================#
        
        # frame_pair_mask, [batch,128*5,128*5]
        # distance, [batch,128*5,128*5]
        # altered_direction, [batch,128*5,128*5,3]
        # orientation [batch,128*5,128*5,3,3]
        for layer in self.layers:
            rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords, side_chain_angles) # add attention
            frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue, aatype_idx)
            x_rigid = layer(x_rigid, altered_direction,  orientation, rigid_mask) # [batch,128, 5, 384]
            x_rigid = self.norm(x_rigid)
            x_rigid = x_rigid+time
            side_chain_angles = self.predict_angles(x_rigid) #[batch,128,4]
        
        return side_chain_angles

