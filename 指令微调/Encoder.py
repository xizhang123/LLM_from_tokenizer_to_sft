import torch
import torch.nn as nn
import torch.nn.functional as F
from Affine import Affine
        
#借来一用，简单改改
class Qwen2RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states#.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
        
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
            self.layer_norm = Qwen2RMSNorm(multi_head_attention.embedding_dim)
        else:
            self.layer_norm = None

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self,query,q_mask):
        #绝对不能用+=，那是原地修改，没法算梯度
        query = query + self.dropout_layer(self.multi_head_attention(query,q_mask,query,self.mask_future,is_cross_attention=False))
        query = query + self.dropout_layer(self.position_wise_feed_forward(query))
        if self.layer_norm is not None:
            query = self.layer_norm(query)
        return query

#编码器
class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers
        
    def forward(self, query, q_mask):
        for encoder_layer in self.encoder_layers:
            query = encoder_layer(query,q_mask)
        return query
