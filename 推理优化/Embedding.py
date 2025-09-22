import math
import torch
import torch.nn as nn
from Affine import Affine

#余弦位置编码
class SinusoidalPE(nn.Module):
    def __init__(self,embedding_dim,max_len):
        super(SinusoidalPE,self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len      = max_len

        # 给位置编码分配空间
        pe=torch.zeros(1,max_len,embedding_dim)

        # 绝对位置信息，max_len * 1
        position = torch.arange(0, max_len).unsqueeze_(1)

        # 词嵌入不同位置的变化速度
        div_term = torch.exp(torch.arange(0,embedding_dim,2) * (-math.log(10000.0) / embedding_dim))

        # 生成余弦位置编码
        pe[...,0::2] = torch.sin(position * div_term)
        pe[...,1::2] = torch.cos(position * div_term)

        # 将位置编码注册为常量
        self.register_buffer('pe',pe)

    def forward(self, x):
        assert x.size(-2) <= self.max_len , "x.size(-2) > self.max_len"
        assert x.size(-1) == self.embedding_dim , "x.size(-1) != self.embedding_dim"
        return x + self.pe[:,:x.size(-2)]

#旋转位置编码
class ROPE(nn.Module):
    def __init__(self,embedding_dim,max_len):
        super(ROPE,self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len      = max_len

        # 给位置编码分配空间
        sin = torch.zeros(1,max_len,embedding_dim//2)
        cos = torch.zeros(1,max_len,embedding_dim//2)
        
        # 绝对位置信息，max_len * 1
        position = torch.arange(0, max_len).unsqueeze_(1)

        # 词嵌入不同位置的变化速度
        div_term = torch.exp(torch.arange(0,embedding_dim,2) * -(math.log(10000.0) / embedding_dim))

        # 获得正弦与余弦部分
        sin[...,:] = torch.sin(position * div_term)
        cos[...,:] = torch.cos(position * div_term)

        # 将位置编码注册为常量
        self.register_buffer('sin',sin)
        self.register_buffer('cos',cos)

    def forward(self, x):
        assert x.size(-2) <= self.max_len , "x.size(-2) > self.max_len"
        assert x.size(-1) == self.embedding_dim , "x.size(-1) != self.embedding_dim"
        x_sin,x_cos = x[...,0::2],x[...,1::2]
        sin = self.sin[:,:x.size(1)]
        cos = self.cos[:,:x.size(1)]
        return torch.cat([ x_sin*cos-x_cos*sin, x_cos*cos+x_sin*sin ],dim=-1)

#自行学习位置编码
class LearnedPE(nn.Module):
    def __init__(self,embedding_dim,max_len):
        super(LearnedPE,self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len      = max_len
        self.pe = nn.Parameter(torch.zeros(1,max_len,embedding_dim))
    def forward(self, x):
        assert x.size(-2) <= self.max_len , "x.size(-2) > self.max_len"
        assert x.size(-1) == self.embedding_dim , "x.size(-1) != self.embedding_dim"
        return x + self.pe[:,:x.size(-2)]

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim,enable_affine,position_information_type,dropout_rate):
        super(Embedding,self).__init__()
        self.look_up_table = nn.Embedding(vocab_size,embedding_dim)
        self.embedding_dim  = embedding_dim
        if enable_affine == True:
            self.affine = Affine(math.sqrt(embedding_dim))
        else:
            self.affine = None
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        if position_information_type == "sinusoidal":
            self.position_encoding_layer = SinusoidalPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "rotary":
            self.position_encoding_layer = ROPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "learned":
            self.position_encoding_layer = LearnedPE(self.embedding_dim,max_len = 5000)
        else:
            self.position_encoding_layer = None
    def forward(self,x):
        x_embed = self.look_up_table(x)
        if self.affine is not None:
            x_embed = self.affine(x_embed)
        else:
            x_embed *= math.sqrt(self.embedding_dim)
        if self.position_encoding_layer is not None:
            x_embed = self.position_encoding_layer(x_embed)
        return self.dropout_layer(x_embed)