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
from foldingdiff import utils
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
        self.mlp2d = nn.Linear(d_model+d_model+d_model+d_model, d_model, dtype=torch.float64)
                
       # self.mlp2d = nn.Sequential(
       #           nn.Linear(d_model+d_model+d_model, d_model//2, dtype=torch.float64),
       #           nn.ReLU(),
       #           nn.Linear(d_model//2, d_model, dtype=torch.float64)
       # )
        self.embed_rigid_idx =  nn.Linear(5, d_model, dtype=torch.float64)

    def forward(
        self, 
        x_seq_esm: torch.Tensor, 
        x_rigid_type: torch.Tensor,
        x_rigid_proterty: torch.Tensor,
    ) -> torch.Tensor: 
        x_embed_rigid_type = self.embed_rigid_type(x_rigid_type) #[batch, L, 5, 128]
        x_embed_rigid_proterty = self.embed_rigid_property(x_rigid_proterty) #[batch, L, 5, 128]
        rigid_idx = torch.zeros([x_rigid_type.shape[0], x_rigid_type.shape[1], 5], dtype=torch.int64)
        rigid_idx[:, :, :] = torch.arange(5) 
        x_rigid_idx = F.one_hot(rigid_idx, num_classes=5)
        x_rigid_idx = x_rigid_idx.double()
        x_embed_rigid_idx = self.embed_rigid_idx(x_rigid_idx)
        x_seq = self.seq2d(x_seq_esm) #[batch, L, 512]
        x_seq = x_seq.unsqueeze(-2)
        input = torch.cat([x_embed_rigid_type, x_embed_rigid_proterty,  x_seq.repeat(1, 1, 5, 1), x_embed_rigid_idx], dim=-1)  #[batch, L, 5, 128*3]
        input_embed = self.mlp2d(input) #[batch, L, 4, 128]
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
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
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
        self.query = nn.Linear(d_model, d_model, dtype=torch.float64)
        self.key = nn.Linear(d_model, d_model, dtype=torch.float64)
        self.value = nn.Linear(d_model, d_model, dtype=torch.float64)
        self.value_3d = nn.Linear(d_model, 3, dtype=torch.float64)
        self.fc = nn.Linear(d_model+3, d_model, dtype=torch.float64)
       # self.mlp =  nn.Sequential(
       #           nn.Linear(16, 8, dtype=torch.float64),
       #           nn.ReLU(),
       #           nn.Linear(8, 1, dtype=torch.float64)
       # )
        self.mlp = nn.Linear(16, 1, dtype=torch.float64)
        
    def forward(self, x_rigid, altered_direction, orientation, attention_mask, distance,  frame_pair_mask=None, seq_correlation_matrix=None, ):
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
        attention_mask = attention_mask.unsqueeze(-3)
        attention_mask = attention_mask.repeat(1, 8, 1, 1)
        if attention_mask is not None:
           # Mask invalid positions
           scores = scores.masked_fill_(attention_mask == 0, -1e9) 
        
        attn_weights = F.softmax(scores, dim=-1) # [batch,8,128*5,128*5] n_heads = 8
        attn_v = torch.matmul(attn_weights, v)   # [batch,8,128*5,128*5]
        attn_v = attn_v.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)  # [batch,128*5,384]
        socres_merged = torch.sum(scores, dim=-3).squeeze(dim=-3)
        attn_weights_merged = F.softmax(socres_merged, dim=-1) # [batch,128*5,128*5]
        
        v3d_map = torch.einsum('bijmn,bjn->bijn', orientation, v_3d)  # [batch,128*5,128*5,3]
        v3d_direction = torch.cross(altered_direction, v3d_map, dim=-1)  # [batch,128*5,128*5,3]
        
        attn_frame = torch.mul(attn_weights_merged.unsqueeze(-1), v3d_direction)  # [batch,128*5,128*5,3] = [batch,128*5,128*5,1] dot [batch,128*5,128*5,3]
        attn_frame = torch.mean(attn_frame,dim=-2)   # [batch,128*5,128*5,3] -> [batch,128*5,3]
        attn_output = torch.cat((attn_v, attn_frame), dim=-1) 
        attn_output = self.fc(attn_output) # [batch,128*5,384+3]->[batch,128*5,3]

        return attn_output

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
        
        x_rigid = self.self_attn(x_rigid, altered_direction, orientation, rigid_mask,distance)
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
# 2. distance未作为尺度加到attenion权重；初步设想，是引入某个投影函数，对distance做数值变化，让其作为attention的约束。rigid的距离太远or太近都不应该发挥太大作用。【类似neighbor思想？】
# 3. bias未加入attention权重计算；
# 4. frame的attention并没有使用多头，而是合并了权重，记得加入多头机制；
# 5. 位置编码需要重新更改；
# 6. 预测角度，MLP的输入是[batch,l,r*5,d] 是否改为[batch,l,r,5,d]
# [batch,seq_len,rigid,d] 

#====================================================最初的版本=================================
'''
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
        
        x_rigid = x_rigid.view(x_rigid.size(0), -1, x_rigid.size(-1)) # [batch,128*5,384]
     #   print("======== x_rigid  view=========")

        for layer in self.layers:
            rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords, side_chain_angles) # add attention
      #      print("========  rigid_by_residue=========")
            frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue, aatype_idx)
       #     print("======== frame_pair_mask, distance, altered_direction, orientation=========")
            x_rigid = layer(x_rigid, altered_direction,  orientation, frame_pair_mask) # [batch,128, 5, 384]
        #    print("========  x_rigid_transformer =========")
            x_rigid = self.norm(x_rigid)
         #   print("========   self.norm(x_rigid) =========")
            x_rigid = x_rigid+time_encoded
          #  print("======== x_rigid+time_encoded =========")            
            x_rigid_1  = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2]//5, 5, x_rigid.shape[-1])
         #   print("======== x_rigid_1 =========")   
            x_rigid_2 = x_rigid.reshape(x_rigid_1.shape[0], x_rigid_1.shape[-3], -1)     
         #   print("======== x_rigid_2 =========")         
            side_chain_angles = self.predict_angles(x_rigid_2) #[batch,128,4]
         #   print("======== side_chain_angle =========") 
        return side_chain_angles
'''
    #====================================================最初的版本=================================
    
'''  
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


        x_rigid = self.pos_encoding(x_rigid) #rigid finish # [batch,128,5,384]
        
        x_rigid = x_rigid.view(x_rigid.size(0), -1, x_rigid.size(-1)) # [batch,128*5,384]
        
        x_rigid = x_rigid+time_encoded ####注意看一下维度变化
        
        rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, backbone_coords, side_chain_angles) # add attention
        frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue, aatype_idx)

        for layer in self.layers:
            x_rigid = layer(x_rigid, altered_direction,  orientation, frame_pair_mask) # [batch,128, 5, 384]
            x_rigid = self.norm(x_rigid)

    
        x_rigid_1  = x_rigid.reshape(x_rigid.shape[0], x_rigid.shape[-2]//5, 5, x_rigid.shape[-1])
        x_rigid_2 = x_rigid.reshape(x_rigid_1.shape[0], x_rigid_1.shape[-3], -1)     
        side_chain_angles = self.predict_angles(x_rigid_2) #[batch,128,4]
        return side_chain_angles
''' 
class Ridge_Transformer(nn.Module):
    def __init__(self,
        d_model: int = 384,
        d_esm_seq: int = 320,
        n_rigid_type: int = 20,
        n_rigid_property: int = 6,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512, #hidden layer dim
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
            chain_angles = self.predict_angles(x_rigid_2) #[batch,128,4]
            
            #side_chain_angles = utils.modulo_with_wrapped_range(side_chain_angles, -torch.pi, torch.pi)
            side_chain_angles = torch.remainder(chain_angles + torch.pi, 2 * torch.pi) - torch.pi
            #print("================side_chain_angles====================",side_chain_angles)
            
        noise = self.predict_noise(x_rigid_2,x_rigid_init)
        return noise
    
        #self.angleT = nn.Linear(d_angles, d_model, dtype=torch.float64)
        #noise_angles = self.angleT(side_chain_angles) #[batch,128,384]
        #noise_angles = noise_angles+time_encoded