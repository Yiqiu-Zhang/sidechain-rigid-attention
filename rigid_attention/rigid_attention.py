import torch 
import torch.nn as nn

class Structure_Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Structure_Attention,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.value_3d = nn.Linear(input_dim,3)
    
    def forward(self, x, frame, pair_seq):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        v_3d = self.value_3d(x)
        scores = torch.matmul(q,k,transpose(-2,-1)) # 要修改记得
        weights = nn.functional.softmax(scores, dim=-1)
        O1 = torch.matmul(weights, v)
        O2 = torch.matmul(weights, pair_seq)
        O3 = torch.matmul(weights, frame[]*frame[]*v_3d)
        return context
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([Structure_Attention(input_dim, output_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(output_dim * num_heads, input_dim)

    def forward(self, x):
        attentions = [attention(x) for attention in self.attentions]
        concatenated = torch.cat(attentions, dim=-1)
        projected = self.projection(concatenated)
        return projected