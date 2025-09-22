import torch
import torch.nn as nn
import torch.nn.functional as F
from Affine import Affine
#层归一化
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(embedding_dim))
        self.b2 = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a2 * (x-mean) / (std+self.eps) + self.b2
        
#针对每个词嵌入的前馈网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self,embedding_dim,feed_forward_dim,enable_affine):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(embedding_dim, feed_forward_dim, bias=False)
        self.w2 = nn.Linear(feed_forward_dim, embedding_dim, bias=False)
        self.enable_affine = enable_affine
        if enable_affine:
            self.a1 = Affine(1.0)
            self.a2 = Affine(1.0)
        
    def forward(self, x):
        if self.enable_affine:
            x = F.relu(self.w1(self.a1(x)))
            return F.relu(self.w2(self.a2(x)))
        else:
            x = F.relu(self.w1(x))
            return F.relu(self.w2(x))

#编码器层
class EncoderLayer(nn.Module):
    def __init__(self,multi_head_attention,mask_future,position_wise_feed_forward,enable_layer_norm,dropout_rate):
        super(EncoderLayer,self).__init__()
        self.multi_head_attention = multi_head_attention
        self.position_wise_feed_forward = position_wise_feed_forward
        self.mask_future = mask_future
        if enable_layer_norm == True:
            self.layer_norm = LayerNorm(multi_head_attention.embedding_dim)
        else:
            self.layer_norm = None

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self,query,q_mask):
        if self.layer_norm is not None:
            query = self.layer_norm(query)
        #绝对不能用+=，那是原地修改，没法算梯度
        query = query + self.dropout_layer(self.multi_head_attention(query,q_mask,query,self.mask_future,is_cross_attention=False))
        return  query + self.dropout_layer(self.position_wise_feed_forward(query))

#编码器
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers
        
    def forward(self, query, q_mask):
        for encoder_layer in self.encoder_layers:
            query = encoder_layer(query,q_mask)
        return query
