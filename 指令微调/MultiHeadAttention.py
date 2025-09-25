import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Affine import Affine

#获取相对位置矩阵
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
    return rela_dist.astype(np.float32)

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

#标记一个需要多次使用的tensor
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

#缓存字典与定时器
reg_dict = dict()
reg_timer = dict()

#查看是否未注册
def un_reg(p):
    return not p in reg_dict

#注册需要重复使用的tensor
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

#从缓冲区中获取可重复使用的张量
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
            scores.masked_fill_(std_mask == 0.0,-1e3)

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
                scores.masked_fill_(std_mask == 0.0,-1e3)
    
            #计算概率权重
            p_attn = F.softmax(scores, dim = -1)
    
            #权重talk
            if talking_after_softmax is not None:
                p_attn = talking_after_softmax(p_attn.transpose(-1,-3)).transpose(-1,-3)
    
            #计算加权求和的结果
            ret[...,i:i+block_size,:] = torch.matmul(p_attn, value_block)
    return ret

#多头注意力
class MultiHeadAttention(nn.Module):
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

        if position_information_type == "mask":
            self.absolute_affine = Affine(1.0,grad_factor=1.0)
            self.relative_affine = Affine(0.1,grad_factor=1.0)
        else:
            self.absolute_affine = None
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