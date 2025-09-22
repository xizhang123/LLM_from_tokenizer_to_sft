import copy
import torch
import torch.nn as nn
from Embedding import Embedding
from MultiHeadAttention import MultiHeadAttention
from DiffMultiHeadAttention import DiffMultiHeadAttention
from Encoder import PositionWiseFeedForward,EncoderLayer,Encoder
from Generator import Projector,Generator

def make_model(vocab_size,embedding_dim,key_dim,head_number,position_information_type,
               enable_affine,enable_talking_head,use_diff,self_attention_block_size,
               feed_forward_dim,enable_layer_norm,deep,dropout_rate,enable_el_cache):
    #嵌入层
    embedding = Embedding(
        vocab_size = vocab_size,
        embedding_dim = embedding_dim,
        enable_affine = enable_affine,
        position_information_type = position_information_type,
        dropout_rate = dropout_rate
    )
    #多头自注意力层
    if use_diff:
        assert use_diff == False, "差分注意力暂未完成EL-Attention的集成，use_diff 应当为 False "
        Attention = DiffMultiHeadAttention
    else:
        Attention = MultiHeadAttention
    assert self_attention_block_size == 0, "暂不兼容EL-Attention与注意力分块，self_attention_block_size 应当为 0 "
    multi_head_attention = Attention(
        embedding_dim = embedding_dim,
        key_dim = key_dim,
        head_number = head_number,
        position_information_type = position_information_type,
        enable_affine = enable_affine,
        enable_talking_head = enable_talking_head,
        self_attention_block_size = self_attention_block_size,
        dropout_rate = dropout_rate,
        enable_el_cache = enable_el_cache
    )
    #信息融合前馈网络
    position_wise_feed_forward = PositionWiseFeedForward(
        embedding_dim = embedding_dim,
        feed_forward_dim = feed_forward_dim,
        enable_affine = enable_affine
    )
    #编码器层
    encoder_layer = EncoderLayer(
        multi_head_attention = copy.deepcopy(multi_head_attention),
        mask_future = True,#自注意力，都要遮盖
        position_wise_feed_forward = copy.deepcopy(position_wise_feed_forward),
        enable_layer_norm = enable_layer_norm,
        dropout_rate = dropout_rate
    )
    #堆叠的编码器层组成编码器
    encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(deep)])
    encoder = Encoder(encoder_layers = encoder_layers)
    #投射器
    projector = Projector(
        embedding_dim = embedding_dim,
        vocab_out_size = vocab_size,
        enable_affine = enable_affine
    )
    #生成器模型本身
    model = Generator(
        embedding = embedding,
        encoder = encoder,
        projector = projector
    )
    #模型参数初始化
    for p in model.parameters():
        #偏置，仿射参数不会随机设置
        #矩阵形式的参数
        if p.dim() == 2:
            nn.init.xavier_uniform_(p)
    return model
    
