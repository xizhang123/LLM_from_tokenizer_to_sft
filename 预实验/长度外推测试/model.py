#注：
#模型的输入为列表，其中可以包含多条等长的序列（不考虑padding），这样设计的目的是保留模型支持多模态的可能
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

#可以使用随机起始位置
use_random = True
random_keeper = 0

#创建模型
def make_model(setting):
    #工具函数
    deep_copy = copy.deepcopy

    #平衡资源消耗与效果
    enable_affine = setting["enable_affine"]
    enable_layer_norm = setting["enable_layer_norm"]
    enable_talking_head = setting["enable_talking_head"]
    enable_gated_linear_unit_in_projector = setting["enable_gated_linear_unit_in_projector"]
    self_attention_block_size = setting["self_attention_block_size"]
    cross_attention_block_size = 0 #先占位，"translator"模式单独更新
    head_number = setting["head_number"]
    if enable_gated_linear_unit_in_projector == True:
        glu_projector_multivalue = setting["glu_projector_multivalue"]
        glu_projector_deep = setting["glu_projector_deep"]

    #模型类型选择
    model_type = setting["model_type"]
    if model_type == "vit":
        summarizer_length = setting["summarizer_length"]
        patch_size = setting["patch_size"]
    if model_type == "classifier":
        summarizer_length = setting["summarizer_length"]
    if model_type == "translator":
        share_embeddings = setting["share_embeddings"]
        cross_attention_block_size = setting["cross_attention_block_size"]
    if model_type == "generator":
        mask_future = True
    else:
        mask_future = False

    #参数规模设置
    embedding_dim = setting["embedding_dim"]
    key_dim = setting["key_dim"]
    assert embedding_dim % head_number == 0 ,"embedding_dim % head_number != 0"
    vocab_in_size_list = setting["vocab_in_size_list"]
    vocab_out_size = setting["vocab_out_size"]
    feed_forward_dim = setting["feed_forward_dim"]
    model_layers_number = setting["model_layers_number"]

    #位置信息添加方式选择
    position_information_type = setting["position_information_type"]

    #增加泛化能力
    dropout_rate = setting["dropout_rate"]

    #创建编码器嵌入层
    if model_type != "vit":
        encoder_embeddings = nn.ModuleList([Embedding(vocab_in_size,embedding_dim,enable_affine,position_information_type,dropout_rate) \
                         for vocab_in_size in vocab_in_size_list])
    else:
        patch_embeddings = nn.ModuleList([PatchEmbedding(patch_size,embedding_dim,enable_affine,position_information_type,dropout_rate)])

    #创建多头注意力层，掩码位置信息需要在计算多头注意力时添加，注意力内部可以使用affine
    multi_head_attention = MultiHeadAttention(embedding_dim,key_dim,head_number,position_information_type, \
                                              enable_affine,enable_talking_head, \
                                              self_attention_block_size, \
                                              cross_attention_block_size, \
                                              dropout_rate)
    
    #创建前馈层
    position_wise_feed_forward = \
    PositionWiseFeedForward(embedding_dim,feed_forward_dim,enable_affine)

    #创建编码器层
    encoder_layer = EncoderLayer(
        deep_copy(multi_head_attention),mask_future,
        deep_copy(position_wise_feed_forward),
        enable_layer_norm,dropout_rate
    )
    
    #创建编码器组层
    encoder_group_layer = EncoderGroupLayer(nn.ModuleList([deep_copy(encoder_layer) for _ in vocab_in_size_list]))
    
    #创建编码器组
    encoder_group = EncoderGroup(nn.ModuleList([deep_copy(encoder_group_layer) for _ in range(model_layers_number)]))
        
    #创建映射器
    if enable_gated_linear_unit_in_projector == True:
        projector = GLUProjector(embedding_dim,vocab_out_size, \
                                 glu_projector_multivalue,glu_projector_deep,enable_affine)
    else:
        projector = Projector(embedding_dim,vocab_out_size,enable_affine)

    #创建模型
    if model_type =="vit":
        #创建总结器
        sequence_summarizer = SequenceSummarizer(embedding_dim,summarizer_length)
        #创建分类器
        model = Vit(
            patch_embeddings     = patch_embeddings,
            encoder_group        = encoder_group,
            sequence_summarizer  = sequence_summarizer,
            projector            = projector
        )
    elif model_type == "classifier":
        #创建总结器
        sequence_summarizer = SequenceSummarizer(embedding_dim,summarizer_length)
        #创建分类器
        model = Classifier(
            encoder_embeddings   = encoder_embeddings,
            encoder_group        = encoder_group,
            sequence_summarizer  = sequence_summarizer,
            projector            = projector
        )
    elif model_type == "generator":
        model = Generator(
            encoder_embeddings = encoder_embeddings,
            encoder_group      = encoder_group,
            projector          = projector
        )
    elif model_type == "translator":
        #创建解码器嵌入层
        decoder_embedding  = Embedding(vocab_out_size,embedding_dim,enable_affine,position_information_type,dropout_rate)
        #尝试进行嵌入层参数共享
        if share_embeddings == True:
            assert vocab_out_size == vocab_in_size_list[0] , "vocab_out_size != vocab_in_size_list[0]"
            encoder_embeddings[0] = decoder_embedding
        #创建解码器层
        decoder_layer = DecoderLayer(
            deep_copy(multi_head_attention),\
            deep_copy(multi_head_attention),\
            deep_copy(position_wise_feed_forward),\
            enable_layer_norm,dropout_rate)
        #创建解码器
        decoder = Decoder(nn.ModuleList([deep_copy(decoder_layer) for _ in \
                       range(model_layers_number)]))
        #创建翻译器模型
        model = Translator(
            encoder_embeddings = encoder_embeddings,
            decoder_embedding  = decoder_embedding,
            encoder_group      = encoder_group,
            decoder            = decoder,
            projector          = projector
        )
    #初始化模型参数
    for p in model.parameters():
        #偏置，仿射参数不会随机设置
        if p.dim() == 1:
            pass
        #矩阵形式的参数
        if p.dim() == 2:
            nn.init.xavier_uniform_(p)
        #门控线性单元阵列因为是特殊运算要单独考虑
        if p.dim() == 3:
            for m in p:
                nn.init.xavier_uniform_(m)
    return model

class PatchEmbedding(nn.Module):
    def __init__(self,patch_size,embedding_dim,enable_affine,position_information_type,dropout_rate):
        super(PatchEmbedding,self).__init__()
        self.img2patch = Rearrange('b c (h p1) (w p2) -> b (h w)(p1 p2 c)',p1=patch_size,p2=patch_size)
        self.patch2vector = nn.Linear(3*patch_size**2,embedding_dim,bias=False)
        self.embedding_dim  = embedding_dim
        if enable_affine == True:
            self.affine = Affine(math.sqrt(embedding_dim))
        else:
            self.affine = None
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        if position_information_type == "sinusoidal_position_encoding":
            self.position_encoding_layer = SinusoidalPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "rotary_position_encoding":
            self.position_encoding_layer = ROPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "rotary_position_encoding_with_random_start":
            self.position_encoding_layer = ROPERS(self.embedding_dim,max_len = 5000,train_start = 1000,test_start = 500)
        elif position_information_type == "position_encoding_learned":
            self.position_encoding_layer = LearnedPE(self.embedding_dim,max_len = 5000)
        else:
            self.position_encoding_layer = None
        
    def forward(self,x):
        x_embed = self.patch2vector(self.img2patch(x))
        if self.affine is not None:
            x_embed = self.affine(x_embed)
        else:
            x_embed *= math.sqrt(self.embedding_dim)
        if self.position_encoding_layer is not None:
            x_embed = self.position_encoding_layer(x_embed)
        return self.dropout_layer(x_embed)

#嵌入层
class Embedding(nn.Module):
    def __init__(self,vocab_in_size,embedding_dim,enable_affine,position_information_type,dropout_rate):
        super(Embedding,self).__init__()
        self.look_up_table = nn.Embedding(vocab_in_size,embedding_dim)
        self.embedding_dim  = embedding_dim
        if enable_affine == True:
            self.affine = Affine(math.sqrt(embedding_dim))
        else:
            self.affine = None
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        if position_information_type == "sinusoidal_position_encoding":
            self.position_encoding_layer = SinusoidalPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "rotary_position_encoding":
            self.position_encoding_layer = ROPE(self.embedding_dim,max_len = 5000)
        elif position_information_type == "rotary_position_encoding_with_random_start":
            self.position_encoding_layer = ROPERS(self.embedding_dim,max_len = 5000,train_start = 1000,test_start = 500)
        elif position_information_type == "position_encoding_learned":
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

#多头注意力
class MultiHeadAttention(nn.Module):
    "多头注意力模块"
    def __init__(self,embedding_dim,key_dim,head_number,position_information_type,enable_affine,enable_talking_head, \
                 self_attention_block_size,cross_attention_block_size,dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim              = embedding_dim
        self.key_dim                    = key_dim
        self.head_number                = head_number
        self.position_information_type  = position_information_type
        self.enable_talking_head        = enable_talking_head
        self.self_attention_block_size  = self_attention_block_size
        self.cross_attention_block_size = cross_attention_block_size
        self.dropout_layer              = nn.Dropout(p=dropout_rate)
        self.enable_affine              = enable_affine

        self.query_w = nn.Linear(embedding_dim,key_dim*head_number,bias=False)
        self.key_w   = nn.Linear(embedding_dim,key_dim*head_number,bias=False)
        self.value_w = nn.Linear(embedding_dim,key_dim*head_number,bias=False)
        self.out_w   = nn.Linear(key_dim*head_number,embedding_dim,bias=False)

        if enable_affine == True:
            self.query_a = Affine(1.0)
            self.key_a   = Affine(1.0)
            self.value_a = Affine(1.0)
            self.out_a   = Affine(1.0)

        if enable_talking_head == True:
            self.talking_before_softmax = nn.Linear(head_number,head_number,bias=False)
            self.talking_after_softmax  = nn.Linear(head_number,head_number,bias=False)
        else:
            self.talking_before_softmax = None
            self.talking_after_softmax  = None

        if position_information_type == "mask_position_information" or position_information_type == "rotary_position_encoding_with_random_start":
            self.absolute_affine = Affine(1.0,grad_factor=1.0)
        else:
            self.absolute_affine = None
        if position_information_type == "mask_position_information":
            self.relative_affine = Affine(0.1,grad_factor=1.0)
        else:
            self.relative_affine = None
        
    def forward(self, query, q_mask, key_value, mask_future, is_cross_attention):
        #经过线性变换得到真正的QKV
        query = self.query_w(query)
        key   = self.key_w(key_value)
        value = self.value_w(key_value)
        
        #进行仿射变换，加快训练速度
        if self.enable_affine == True:
            query = self.query_a(query)
            key   = self.key_a(key)
            value = self.value_a(value)

        #对词嵌入进行划分，准备进行多头注意力计算
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.head_number, self.key_dim).transpose(1,2)
        key   = key.view(batch_size, -1, self.head_number, self.key_dim).transpose(1,2)
        value = value.view(batch_size, -1, self.head_number, self.key_dim).transpose(1,2)
        
        #计算分块多头注意力
        out = attention(query, q_mask, key, value, \
                        self.absolute_affine, self.relative_affine, \
                        self.talking_before_softmax, self.talking_after_softmax, \
                        self_attention_block_size = self.self_attention_block_size, \
                        cross_attention_block_size = self.cross_attention_block_size, \
                        mask_future = mask_future, is_cross_attention = is_cross_attention)
        
        #将计算完注意力的结果拼接回去
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.head_number * self.key_dim)
        if self.enable_affine:
            return self.dropout_layer(self.out_a(self.out_w(out)))
        else:
            return self.dropout_layer(self.out_w(out))

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

    def forward(self,sequence,q_mask):
        if self.layer_norm is not None:
            temp = self.layer_norm(sequence)
        else:
            temp = sequence
        temp = temp + self.dropout_layer(self.multi_head_attention(temp,q_mask,temp,self.mask_future,is_cross_attention=False))
        return  temp + self.dropout_layer(self.position_wise_feed_forward(temp))

#编码器组层
class EncoderGroupLayer(nn.Module):
    def __init__(self,encoder_layers):
        super(EncoderGroupLayer,self).__init__()
        self.encoder_layers = encoder_layers
        
    def forward(self,sequences,q_mask):
        ret = []
        for encoder_layer,sequence in zip(self.encoder_layers,sequences):
            ret += [encoder_layer(sequence,q_mask)]
        return ret

#编码器组
class EncoderGroup(nn.Module):
    def __init__(self, encoder_group_layers):
        super(EncoderGroup, self).__init__()
        self.encoder_group_layers = encoder_group_layers
        
    def forward(self, sequences, q_mask):
        for encoder_group_layer in self.encoder_group_layers:
            sequences = encoder_group_layer(sequences,q_mask)
        ret = torch.zeros_like(sequences[0])
        for sequence in sequences:
            ret += sequence
        return ret

#映射器
class GLUProjector(nn.Module):
    def __init__(self,embedding_dim,vocab_out_size,multivalue,deep,enable_affine):
        super(GLUProjector,self).__init__()
        self.enable_affine = enable_affine
        if enable_affine:
            self.affine = Affine(1.0)
        self.multivalue = multivalue
        self.project = nn.Linear(embedding_dim,vocab_out_size*multivalue,bias=False)
        self.glu_group = GLUGroup(vocab_out_size,multivalue,deep)
    def forward(self, x):
        x = F.relu(self.project(x))
        x = x.reshape(*(x.shape[:-1]),x.shape[-1]//self.multivalue,1,self.multivalue)
        if self.enable_affine:
            out = self.affine(self.glu_group(x))
        else:
            out = self.glu_group(x)
        return F.log_softmax(out,dim=-1)

class Projector(nn.Module):
    def __init__(self,embedding_dim,vocab_out_size,enable_affine):
        super(Projector,self).__init__()
        self.enable_affine = enable_affine
        self.project = nn.Linear(embedding_dim,vocab_out_size,bias=False)
        if enable_affine:
            self.affine = Affine(1.0)
    def forward(self, x):
        if self.enable_affine:
            return F.log_softmax(self.affine(self.project(x)),dim=-1)
        else:
            return F.log_softmax(self.project(x),dim=-1)

#总结器
class SequenceSummarizer(nn.Module):
    def __init__(self,embedding_dim,summarizer_length):
        super(SequenceSummarizer,self).__init__()
        self.embedding_dim = embedding_dim
        self.summarizer = nn.Parameter(torch.zeros(summarizer_length,embedding_dim))
        self.project = nn.Linear(summarizer_length,1,bias=False)
    def forward(self,memory):
        scores = torch.matmul(self.summarizer,memory.transpose(-1,-2))/math.sqrt(self.embedding_dim)
        p_attn = F.softmax(scores, dim = -1)
        return self.project(torch.matmul(p_attn, memory).transpose(-1,-2)).transpose(-1,-2)

#图像分类器
class Vit(nn.Module):
    def __init__(self,patch_embeddings,encoder_group,sequence_summarizer,projector):
        super(Vit, self).__init__()
        self.model_type          = "vit"
        self.patch_embeddings    = patch_embeddings
        self.encoder_group       = encoder_group
        self.sequence_summarizer = sequence_summarizer
        self.projector           = projector

    def forward(self,querys,q_mask):
        global use_random
        use_random = True
        emb_querys = [embedding(query) for embedding,query in zip(self.patch_embeddings,querys)]
        memory = self.encoder_group(emb_querys,q_mask)
        summary = self.sequence_summarizer(memory)
        return self.projector(summary)

#分类器
class Classifier(nn.Module):
    def __init__(self,encoder_embeddings,encoder_group,sequence_summarizer,projector):
        super(Classifier, self).__init__()
        self.model_type          = "classifier"
        self.encoder_embeddings  = encoder_embeddings
        self.encoder_group       = encoder_group
        self.sequence_summarizer = sequence_summarizer
        self.projector           = projector

    def forward(self,querys,q_mask):
        global use_random
        use_random = True
        emb_querys = [embedding(query) for embedding,query in zip(self.encoder_embeddings,querys)]
        memory = self.encoder_group(emb_querys,q_mask)
        summary = self.sequence_summarizer(memory)
        return self.projector(summary)

#生成器，生成器不支持多模态
class Generator(nn.Module):
    def __init__(self,encoder_embeddings,encoder_group,projector):
        super(Generator, self).__init__()
        self.model_type         = "generator"
        self.encoder_embeddings = encoder_embeddings
        self.encoder_group      = encoder_group
        self.projector          = projector

    def forward(self,query,q_mask):
        global use_random
        use_random = True
        emb_query = self.encoder_embeddings[0](query)
        out = self.encoder_group([emb_query],q_mask)
        return self.projector(out)

#解码器层
class DecoderLayer(nn.Module):
    def __init__(self,self_multi_head_attention,cross_multi_head_attention,position_wise_feed_forward,enable_layer_norm,dropout_rate):
        super(DecoderLayer,self).__init__()
        self.self_multi_head_attention = self_multi_head_attention
        self.cross_multi_head_attention = cross_multi_head_attention
        self.position_wise_feed_forward = position_wise_feed_forward
        if enable_layer_norm == True:
            self.layer_norm_answer = LayerNorm(self_multi_head_attention.embedding_dim)
            self.layer_norm_memory = LayerNorm(cross_multi_head_attention.embedding_dim)
        else:
            self.layer_norm_answer = None
            self.layer_norm_memory = None

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self,answer,a_mask,memory):
        if self.layer_norm_answer is not None and self.layer_norm_memory is not None:
            temp_answer = self.layer_norm_answer(answer)
            temp_memory = self.layer_norm_memory(memory)
        else:
            temp_answer = answer
            temp_memory = memory

        temp_answer = temp_answer + self.dropout_layer(self.self_multi_head_attention(temp_answer,a_mask,temp_answer,mask_future=True,is_cross_attention=False))
        temp_answer = temp_answer + self.dropout_layer(self.cross_multi_head_attention(temp_answer,a_mask,temp_memory,mask_future=False,is_cross_attention=True))
        return temp_answer + self.dropout_layer(self.position_wise_feed_forward(temp_answer))

#解码器
class Decoder(nn.Module):
    def __init__(self,decoder_layers):
        super(Decoder,self).__init__()
        self.decoder_layers = decoder_layers
        
    def forward(self, answer, a_mask, memory):
        for decoder_layer in self.decoder_layers:
            answer = decoder_layer(answer,a_mask,memory)
        return answer

#翻译器
class Translator(nn.Module):
    def __init__(self,encoder_embeddings,decoder_embedding,encoder_group,decoder,projector):
        super(Translator, self).__init__()
        self.model_type         = "translator"
        self.decoder_embedding  = decoder_embedding
        self.encoder_embeddings = encoder_embeddings
        self.encoder_group      = encoder_group
        self.decoder            = decoder
        self.projector          = projector

    def forward(self,querys,q_mask,answer,a_mask):
        global use_random
        use_random = True
        emb_querys = [embedding(query) for embedding,query in zip(self.encoder_embeddings,querys)]
        emb_answer = self.decoder_embedding(answer)
        memory = self.encoder_group(emb_querys,q_mask)
        out = self.decoder(emb_answer,a_mask,memory)
        return self.projector(out)

#仿射变换
class Affine(nn.Module):
    def __init__(self,value,grad_factor=100):
        super(Affine,self).__init__()
        self.value = nn.Parameter(torch.ones(1)*value/grad_factor)
        self.bias = nn.Parameter(torch.zeros(1))
        self.grad_factor=grad_factor
        
    def forward(self, x):
        return (x*self.value+self.bias)*self.grad_factor

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

class ROPERS(nn.Module):
    def __init__(self,embedding_dim,max_len = 5000,train_start = 1000,test_start = 500):
        super(ROPERS,self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len      = max_len
        self.train_start  = train_start
        self.test_start   = test_start

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
        global use_random
        global random_keeper
        if use_random:
            use_random = False
            if self.training:
                start = random.randint(0,self.train_start)
            else:
                start = random.randint(0,self.test_start)
            random_keeper = start
        else:
            start = random_keeper
        assert x.size(-2)+start <= self.max_len , "x.size(-2)+start > self.max_len"
        assert x.size(-1) == self.embedding_dim , "x.size(-1) != self.embedding_dim"
        x_sin,x_cos = x[...,0::2],x[...,1::2]
        sin = self.sin[:,start:start+x.size(1)]
        cos = self.cos[:,start:start+x.size(1)]
        return torch.cat([ x_sin*cos-x_cos*sin, x_cos*cos+x_sin*sin ],dim=-1)

reg_dict = dict()
reg_timer = dict()
def un_reg(p):
    return not p in reg_dict
def reg(p,v):
    #找缓冲中用的最少的
    keys = [k for k in reg_dict]
    time_min = 0
    if len(keys) != 0:
        key_min = keys[0]
        time_min = reg_timer[key_min]
        for k in keys:
            if reg_timer[k]<time_min:
                key_min = k
                time_min = reg_timer[key_min]
    #计数
    if not p in reg_timer:
        reg_timer[p] = 1
    else:
        reg_timer[p] += 1
    #缓冲满了就删掉最少用的
    if len(keys) > 12:
        del reg_dict[key_min]
    #比最小的值大就保留
    if reg_timer[p] > time_min or len(keys) < 12:
        reg_dict[p] = v
    
def get_reg(p):
    reg_timer[p] += 1
    return reg_dict[p]

#注意力运算
def attention(query, q_mask, key, value, \
              absolute_affine, relative_affine, \
              talking_before_softmax, talking_after_softmax, \
              self_attention_block_size, cross_attention_block_size, \
              mask_future, is_cross_attention):
    #cross_attention有特殊的分块方式
    if is_cross_attention:
        block_size = cross_attention_block_size
    else:
        block_size = self_attention_block_size
        
    #获取词嵌入大小，用于缩放点积运算
    query_dim = query.size(-1)
    #提前调整q_mask的形状，方便广播
    #query:[batch,head,query_len,emb_dim]
    #q_mask:[batch,query_len]
    #q_mask:[batch,query_len]->[batch,1,query_len]
    #q_mask:[batch,1,query_len]->[batch,head,query_len]
    q_mask = q_mask.unsqueeze(1).expand(*(query.size()[:-1]))
    #判断是否需要分块运算
    if block_size == 0:
        #不进行分块
        #计算scores
        scores = torch.matmul(query,key.transpose(-1,-2))/math.sqrt(query_dim)
        
        #尝试添加相对位置信息
        if relative_affine is not None:
            p = ident(['r',0,0,0,query.size(-2),key.size(-2),is_cross_attention])
            if un_reg(p):
                rela_dist = get_relative_dist(0,0,0,query.size(-2),key.size(-2),is_cross_attention)
                #直接广播更高效
                rela_dist = torch.from_numpy(rela_dist).detach().to(query.device)
                reg(p,rela_dist)
            else:
                rela_dist = get_reg(p)
            dist_decay= rela_dist.mul(relative_affine(1.0)).add(1.0).reciprocal()
            scores    = scores.mul(dist_decay)
            
        #尝试添加绝对位置信息
        if absolute_affine is not None:
            p = ident(['a',0,0,0,query.size(-2),key.size(-2),is_cross_attention])
            if un_reg(p):
                abs_mask  = get_absolute_mask(0,0,0,query.size(-2),key.size(-2),is_cross_attention)
                #mask:[query_len,key_len]->[batch,head,query_len,key_len]
                abs_mask  = torch.from_numpy(abs_mask).unsqueeze_(0).unsqueeze_(0).detach().to(query.device)
                reg(p,abs_mask)
            else:
                abs_mask  = get_reg(p)
            abs_mask  = abs_mask.expand(*(scores.size()))
            value_to_sub = absolute_affine(1.0)
            scores = torch.where(abs_mask == 0, scores - value_to_sub, scores)

        #遮挡信息之前先talk，这样数值稳定
        if talking_before_softmax is not None:
            scores = talking_before_softmax(scores.transpose(-1,-3)).transpose(-1,-3)
        
        #是否需要遮挡未来信息
        if mask_future == True:
            p = ident(['f',0,0,0,query.size(-2),key.size(-2),is_cross_attention])
            if un_reg(p):
                #创建遮挡未来信息的掩码
                #mask:[query_len,key_len]->[batch,head,query_len,key_len]
                std_mask = get_std_mask(0,0,0,query.size(-2),key.size(-2),is_cross_attention)
                std_mask = torch.from_numpy(std_mask).unsqueeze_(0).unsqueeze_(0).detach().to(query.device)
                reg(p,std_mask)
            else:
                std_mask = get_reg(p)
            std_mask = std_mask.expand(*(scores.size()))
            #q_mask:[batch,head,query_len]->[batch,head,query_len,key_len]
            std_mask = q_mask.unsqueeze_(-1).expand(*(std_mask.size())) & std_mask
            scores.masked_fill_(std_mask == 0.0,-1e13)

        #计算概率权重
        p_attn = F.softmax(scores, dim = -1)

        #权重talk
        if talking_after_softmax is not None:
            p_attn = talking_after_softmax(p_attn.transpose(-1,-3)).transpose(-1,-3)

        #计算加权求和的结果
        ret = torch.matmul(p_attn, value)
    else:
        #分块时需要一个空间存放最终计算结果
        ret = torch.zeros_like(query)
        #分块操作
        for i in range(0,query.size(-2),block_size):
            #进行分块
            query_block  =  query[...,i:i+block_size,:]
            q_mask_block = q_mask[...,i:i+block_size]
            if is_cross_attention:
                key_block    =    key
                value_block  =  value
            else:
                key_block    =    key[...,max(0,i-block_size):i+block_size*2,:]
                value_block  =  value[...,max(0,i-block_size):i+block_size*2,:]
            #计算scores
            scores = torch.matmul(query_block,key_block.transpose(-1,-2))/math.sqrt(query_dim)
            
            #尝试添加相对位置信息
            if relative_affine is not None: 
                p = ident(['r',i,i,block_size,query.size(-2),key.size(-2),is_cross_attention])
                if un_reg(p):
                    rela_dist = get_relative_dist(i,i,block_size,query.size(-2),key.size(-2),is_cross_attention)
                    rela_dist = torch.from_numpy(rela_dist).detach().to(query.device)
                    reg(p,rela_dist)
                else:
                    rela_dist = get_reg(p)
                # dist_decay= 1.0 / (1 + rela_dist*relative_affine(1.0))
                dist_decay= rela_dist.mul(relative_affine(1.0)).add(1.0).reciprocal()
                scores    = scores.mul(dist_decay)
                
            #尝试添加绝对位置信息
            if absolute_affine is not None:
                p = ident(['a',i,i,block_size,query.size(-2),key.size(-2),is_cross_attention])
                if un_reg(p):
                    abs_mask  = get_absolute_mask(i,i,block_size,query.size(-2),key.size(-2),is_cross_attention)
                    abs_mask  = torch.from_numpy(abs_mask).unsqueeze_(0).unsqueeze_(0).detach().to(query.device)
                    reg(p,abs_mask)
                else:
                    abs_mask = get_reg(p)
                abs_mask  = abs_mask.expand(*(scores.size()))
                value_to_sub = absolute_affine(1.0)
                scores = torch.where(abs_mask == 0, scores - value_to_sub, scores)
    
            #遮挡信息之前先talk，这样数值稳定
            if talking_before_softmax is not None:
                scores = talking_before_softmax(scores.transpose(-1,-3)).transpose(-1,-3)
                
            #是否需要遮挡未来信息
            if mask_future == True:
                p = ident(['f',i,i,block_size,query.size(-2),key.size(-2),is_cross_attention])
                if un_reg(p):
                    #创建遮挡未来信息的掩码，因为是批次操作，需要进行升维
                    std_mask = get_std_mask(i,i,block_size,query.size(-2),key.size(-2),is_cross_attention)
                    std_mask = torch.from_numpy(std_mask).unsqueeze_(0).unsqueeze_(0).detach().to(query.device)
                    reg(p,std_mask)
                else:
                    std_mask = get_reg(p)
                std_mask = std_mask.expand(*(scores.size()))
                std_mask = q_mask_block.unsqueeze(-1).expand(*(std_mask.size())) & std_mask
                scores.masked_fill_(std_mask == 0.0,-1e13)
    
            #计算概率权重
            p_attn = F.softmax(scores, dim = -1)
    
            #权重talk
            if talking_after_softmax is not None:
                p_attn = talking_after_softmax(p_attn.transpose(-1,-3)).transpose(-1,-3)
    
            #计算加权求和的结果
            ret[...,i:i+block_size,:] = torch.matmul(p_attn, value_block)
    return ret

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

#GLU组
class GLUGroup(nn.Module):
    def __init__(self,vocab_out_size,multi_value,deep):
        super(GLUGroup,self).__init__()
        self.deep = deep
        self.wu = nn.ParameterList([nn.Parameter(torch.zeros(vocab_out_size,multi_value,multi_value)) for _ in range(deep)])
        self.wv = nn.ParameterList([nn.Parameter(torch.zeros(vocab_out_size,multi_value,multi_value)) for _ in range(deep)])
        self.wo = nn.ParameterList([nn.Parameter(torch.zeros(vocab_out_size,multi_value,multi_value)) for _ in range(deep-1)]+
                  [nn.Parameter(torch.zeros(vocab_out_size,multi_value,1))])
        self.affine = nn.ModuleList([Affine(1) for _ in range(deep)])
        
    def forward(self,x):
        for i in range(self.deep):
            x = self.affine[i](x)
            u = F.relu(x@self.wu[i])
            v = F.relu(x@self.wv[i])
            x = (u*v)@self.wo[i]
            if i < self.deep-1:
                x = F.relu(x)
        return x.reshape(*(x.shape[:-2]))
        
def get_relative_mat(height,width,k=0):
    posi_i = np.arange(k,height+k) #列的范围
    posi_j = np.arange(0,width) #行的范围
    posi_grid = np.meshgrid(posi_i, posi_j, indexing='ij')
    return abs(posi_grid[0]-posi_grid[1])
    
#用于添加绝对位置信息的掩码
def get_relative_dist(i,j,block_size,i_end,j_end,is_cross_attention):
    if block_size == 0:
        assert i==0 and j==0 ,"i!=0 or j!=0"
        return get_relative_mat(i_end,j_end,k=0)
    if is_cross_attention:
        return get_relative_mat(min(block_size,i_end-i),j_end,k=i)
    #i,j:当前分块的起始位置
    #block_size:分块大小
    #i_end,j_end:序列的长度
    height = block_size #高度，也就是第一个序列中截取的长度，与分块大小相等
    width  = block_size * 3 #宽度，也就是第二个序列中截取的长度，为了更长的上下文，还需要考虑上一个分块和下一个分块
    #创建用来遮挡未来信息的标准掩码
    #i越大，可见的部分越多，j相反，+block_size是因为上一个分块可见。
    rela_dist = get_relative_mat(height,width,k=block_size+i-j)
    #边界超出处理
    #下超出
    down_out = max(0,i+height-i_end)
    #左超出
    left_out = max(0,block_size-j)
    #右超出
    right_out = max(0,j+block_size*2-j_end)
    #边界内截取
    rela_dist = rela_dist[:height-down_out,left_out:width-right_out]
    return rela_dist.astype('float32')

#用于添加绝对位置信息的掩码
def get_absolute_mask(i,j,block_size,i_end,j_end,is_cross_attention):
    if block_size == 0:
        assert i==0 and j==0 ,"i!=0 or j!=0"
        return np.triu(np.ones((i_end,j_end),dtype='bool'), k=0)
    if is_cross_attention:
        return np.triu(np.ones((min(block_size,i_end-i),j_end),dtype='bool'), k=i)
    #i,j:当前分块的起始位置
    #block_size:分块大小
    #i_end,j_end:序列的长度
    height = block_size #高度，也就是第一个序列中截取的长度，与分块大小相等
    width  = block_size * 3 #宽度，也就是第二个序列中截取的长度，为了更长的上下文，还需要考虑上一个分块和下一个分块
    #创建用来遮挡未来信息的标准掩码
    #i越大，可见的部分越多，j相反，+block_size是因为上一个分块可见。
    abs_mask = np.triu(np.ones((height,width),dtype='bool'), k=block_size+i-j)
    #边界超出处理
    #下超出
    down_out = max(0,i+height-i_end)
    #左超出
    left_out = max(0,block_size-j)
    #右超出
    right_out = max(0,j+block_size*2-j_end)
    #边界内截取
    abs_mask = abs_mask[:height-down_out,left_out:width-right_out]
    return abs_mask

#用于遮挡未来信息的标准掩码
def get_std_mask(i,j,block_size,i_end,j_end,is_cross_attention):
    if block_size == 0:
        assert i==0 and j==0 ,"i!=0 or j!=0"
        return np.triu(np.ones((i_end,j_end),dtype='bool'), k=1) == False
    if is_cross_attention:
        return np.triu(np.ones((min(block_size,i_end-i),j_end),dtype='bool'), k=1+i) == False
    #i,j:当前分块的起始位置
    #block_size:分块大小
    #i_end,j_end:序列的长度
    height = block_size #高度，也就是第一个序列中截取的长度，与分块大小相等
    width  = block_size * 3 #宽度，也就是第二个序列中截取的长度，为了更长的上下文，还需要考虑上一个分块和下一个分块
    #创建用来遮挡未来信息的标准掩码
    #i越大，可见的部分越多，j相反，+block_size是因为上一个分块可见。
    std_mask = np.triu(np.ones((height,width),dtype='bool'), k=1+block_size+i-j)
    #边界超出处理
    #下超出
    down_out = max(0,i+height-i_end)
    #左超出
    left_out = max(0,block_size-j)
    #右超出
    right_out = max(0,j+block_size*2-j_end)
    #边界内截取
    std_mask = std_mask[:height-down_out,left_out:width-right_out]
    return std_mask == False

def ident(p_list):
    i,j,block_size,i_end,j_end,is_cross_attention = p_list[1:]
    ret = [p_list[0]]
    if p_list[0]=='r' or p_list[0]=='a':
        if block_size == 0:
            ret += [i_end,j_end,0]
        elif is_cross_attention:
            ret += [min(block_size,i_end-i),j_end,i]
        else:
            height = block_size
            width  = block_size * 3
            ret += [height,width,block_size+i-j]
            down_out = max(0,i+height-i_end)
            left_out = max(0,block_size-j)
            right_out = max(0,j+block_size*2-j_end)
            ret += [height-down_out,left_out,width-right_out]
    else:
        if block_size == 0:
            ret += [i_end,j_end,1]
        elif is_cross_attention:
            ret += [min(block_size,i_end-i),j_end,1+i]
        else:
            height = block_size
            width  = block_size * 3
            ret += [height,width,1+block_size+i-j]
            down_out = max(0,i+height-i_end)
            left_out = max(0,block_size-j)
            right_out = max(0,j+block_size*2-j_end)
            ret += [height-down_out,left_out,width-right_out]
    return str(ret)