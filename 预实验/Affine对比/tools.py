import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import use_random

#用于训练的数据对象
class Batch:
    def __init__(self,input_sequences,target = None):
        #有可能是生成式模型，只有一个序列，所以target默认为None
        #编码器的输入序列，为了支持多模态，所以是列表
        #但暂时没有多模态的需求，所以默认列表中只有一个序列或者是一些等长的序列（padding不算序列长度）
        if input_sequences[0].dim() == 4:
            self.data_type = "vit"
            self.querys = input_sequences
            self.q_mask = torch.from_numpy(np.ones((input_sequences[0].size(0),196),dtype='bool')).to(input_sequences[0].device)
            #分类标签
            self.label = target
            self.ntokens = input_sequences[0].size(0)
        elif target is not None and target.size(-1) > 1:
            #这里是翻译任务
            self.data_type = "translator"
            #获取源语言序列
            self.querys = input_sequences
            #计算源语言序列的掩码
            self.q_mask = input_sequences[0] != 0
            #起始标记+当前已知目标语言序列（并行）
            self.answer  = target[:,:-1]
            #当前已知目标语言序列（并行）掩码
            self.a_mask = (self.answer != 0)
            #标签
            self.label = target[:,1:]
            #完成推理产生的token数量
            self.ntokens = float((self.label != 0).sum())
        elif target is not None and target.size(-1) == 1:
            #这里是分类任务
            self.data_type = "classifier"
            #输入序列和掩码
            self.querys = input_sequences
            self.q_mask = input_sequences[0] != 0
            #分类标签
            self.label = target
            self.ntokens = input_sequences[0].size(0)
        else:
            #这里是生成任务
            self.data_type = "generator"
            self.query = input_sequences[0][:-1]
            self.q_mask = self.query != 0
            self.label = input_sequences[0][1:]
            self.ntokens = float((self.label != 0).sum())

class LossRecorder:
    def __init__(self):
        self.warm_up_step=-1
        self.mean_loss=-1.0
        self.history_decay=0.9
        self.new_loss_weight=1-self.history_decay
        self.train_loss_line_per_epoch = []
        self.test_loss_line_per_epoch = []
        self.train_loss_line = []
        self.test_loss_line = []
        self.mean_loss_line = []
    
    def update_mean_loss_line(self, new_loss):
        "记录损失函数值"
        if self.mean_loss < 0.0:
            self.mean_loss = new_loss
        else:
            self.mean_loss *= self.history_decay
            self.mean_loss += new_loss * self.new_loss_weight
            self.mean_loss_line += [self.mean_loss]
            
#获得标准优化器
def get_std_opt(model,fast=False,lr_rate=1e-6):
    #可以选择使用更高效的官方adam还是使用更可控的简单实现
    if fast == True:
        return LinearRestartOptimizerWrapper(   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), lr_rate)
    else:
        return LinearRestartOptimizerWrapper(SimpleAdamOptimizer(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), lr_rate)

#带有标签平滑的交叉熵损失函数（一般标签平滑作用不好，但仍然保留用于实验）
class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, output_vocab_size, smoothing=0.0):
        super(CrossEntropyLossWithLabelSmoothing, self).__init__()
        # 使用KL散度损失函数（接受对数概率分布x和概率分布y,并不是简单的KL散度计算）
        self.criterion = nn.KLDivLoss(reduction='sum')
        #平滑的结果与输出词表大小相关
        self.output_vocab_size = output_vocab_size
        #标签概率不再是1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, model_output_dist, target_sequence):
        #根据模型输出的分布与标签的分布计算交叉熵损失
        #两个概率分布的大小必须相同相同
        assert model_output_dist.size(1) == self.output_vocab_size
        #为目标分布分配和模型输出形状、类型一样的空间，默认不追踪梯度，写明更清晰
        true_dist = torch.zeros_like(model_output_dist,requires_grad=False)
        #将概率匀给其他位置，挖去0、1、自身
        true_dist.fill_(self.smoothing / (self.output_vocab_size - 3))
        #使用置信度填充目标词的位置（true_dist是词表那么长的概率分布）
        #目标序列升维，target_sequence:[batch*len]->[batch*len,1]
        #true_dist:[batch*len,vocab]
        #在vocab的维度上用标签值当作索引，找到对应元素，填充self.confidence
        true_dist.scatter_(1, target_sequence.data.unsqueeze(1), self.confidence)
        #将填充位置和起始位置概率设为0
        true_dist[:,0] = 0
        true_dist[:,1] = 0
        #目标序列中等于0的元素的坐标
        #因为target_sequence形状为[batch*len],所以torch.nonzero(...)是只有一列的二维张量
        #消除第二个维度，成为一维张量
        padding_index_for_target_sequence = torch.nonzero(target_sequence.data == 0).squeeze(1)
        #true_dist:[batch*len,vocab]，padding_index_for_target_sequence的数值范围也在batch*len
        #被填充的位置，整个词表那么长的概率分布变成全0.0
        true_dist.index_fill_(0, padding_index_for_target_sequence, 0.0)
        #计算模型输出分布与平目标序列标签平滑后的分布之间的交叉熵
        #model_output_dist是对数概率分布,由F.log_softmax(self.project(x),dim=-1)产生
        return self.criterion(model_output_dist, true_dist)

#训练一个epoch
def run_epoch(model,data_iter,weight_updater):
    #记录开始时间、生成的token、累积的loss
    time_start = time.time()
    accumulate_tokens = 0
    accumulate_loss = 0.0
    current_accumulate_tokens = 0
    current_accumulate_loss = 0.0
    for step, batch in enumerate(data_iter):
        assert model.model_type == batch.data_type, model.model_type+" != "+batch.data_type
        if model.model_type == "translator":
            model_output = model.forward(batch.querys, batch.q_mask, batch.answer, batch.a_mask)
        elif model.model_type == "classifier" or model.model_type == "vit":
            model_output = model.forward(batch.querys, batch.q_mask)
        elif model.model_type == "generator":
            model_output = model.forward(batch.query, batch.q_mask)
        #计算损失并进行梯度下降
        loss_per_batch = weight_updater(model_output, batch.label, batch.ntokens, model.training)
        #累积token数量
        accumulate_tokens += batch.ntokens
        current_accumulate_tokens += batch.ntokens
        #累积loss
        accumulate_loss += loss_per_batch
        current_accumulate_loss += loss_per_batch
        #一个统计周期结束，打印统计结果
        if step % 50 == 0:
            consumed_time = time.time() - time_start
            print("Step: %d Loss: %f Tokens per Sec: %f" %
                    (step, current_accumulate_loss/current_accumulate_tokens, \
                     current_accumulate_tokens/consumed_time))
            time_start = time.time()
            current_accumulate_tokens = 0
            current_accumulate_loss = 0.0
    #返回一个epoch的统计信息
    loss_per_epoch = accumulate_loss / accumulate_tokens
    if model.training:
        weight_updater.recorder.train_loss_line_per_epoch+=[loss_per_epoch]
    else:
        weight_updater.recorder.test_loss_line_per_epoch+=[loss_per_epoch]
    return loss_per_epoch

#计算损失函数并进行一步梯度下降
class LossComputeAndStep:
    def __init__(self, criterion, optimizer_wrapper):
        self.criterion         = criterion
        self.optimizer_wrapper = optimizer_wrapper
        self.recorder          = optimizer_wrapper.recorder
        
    def __call__(self, model_output, label, ntokens, is_training):
        model_output_dist = model_output.view(-1,model_output.size(-1))
        true_dist = label.reshape(-1)
        loss = self.criterion(model_output_dist, true_dist)
        loss_value = float(loss)
        if(is_training):
            loss.backward()                                 #反向传播
            self.optimizer_wrapper.step()                   #梯度下降
            self.optimizer_wrapper.optimizer.zero_grad()    #梯度归零
            self.recorder.update_mean_loss_line(loss_value/ntokens)
            self.recorder.train_loss_line+=[loss_value/ntokens]
        else:
            self.recorder.test_loss_line+=[loss_value/ntokens]
        return loss_value

class LinearRestartOptimizerWrapper:
    def __init__(self, optimizer, lr_rate):
        self.optimizer = optimizer         #优化器,用于执行梯度下降
        self.lr_rate = lr_rate             #学习率增长率速度
        self.recorder = LossRecorder()     #损失记录器
        self.total_step = 1                #总步数
        self.start_step = 0                #上次学习率重启后经过的步数

    def step(self):
        #设置优化器中每个参数组的学习率并执行梯度下降
        lrate = self.lrate()
        for parameters in self.optimizer.param_groups:
            parameters['lr'] = lrate
        self.optimizer.step()
        #定期打印步数与学习率，便于调试
        if self.total_step > 0 and self.total_step %50==0:
            print("total_step:",self.total_step,"lr:",lrate)
        
    def lrate(self):
        inc_steps = self.total_step-self.start_step+1
        if inc_steps >= max(100,self.recorder.warm_up_step) and self.recorder.mean_loss > self.recorder.mean_loss_line[-(inc_steps//2)]:
            self.recorder.warm_up_step = inc_steps*9//10
            print("lr restart at step: %d, lr: %f, new_loss: %f > mid_loss: %f" % \
                  (self.total_step, self.lr_rate * inc_steps, self.recorder.mean_loss, self.recorder.mean_loss_line[-(inc_steps//2)]))
            self.start_step = self.total_step
            inc_steps = 1
        self.total_step += 1
        return self.lr_rate * inc_steps

class SimpleAdamOptimizer:
    "简单的自适应矩估计优化器"
    def __init__(self, params, lr, betas, eps):
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = eps
        self.t = 0
        self.param_groups = []
        for p in params:
            self.param_groups.append({'params': p, 'lr': lr, 'm': None, 'v': None})

    def step(self):
        self.t += 1
        for group in self.param_groups:
            if group['m'] is None:
                group['m'] = torch.zeros_like(group['params'])
                group['v'] = torch.zeros_like(group['params'])

            # Compute gradients
            grad = group['params'].grad
            if grad is None:
                continue

            group['m'].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            group['v'].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # Bias correction
            m_hat = group['m'] / (1 - self.beta1 ** self.t)
            v_hat = group['v'] / (1 - self.beta2 ** self.t)

            # Update parameters
            with torch.no_grad():
                group['params'].sub_(group['lr'] / (v_hat.sqrt() + self.epsilon) * m_hat)

    def zero_grad(self):
        for group in self.param_groups:
            if group['params'].grad is not None:
                group['params'].grad.detach_()
                group['params'].grad.zero_()

    
def greedy_decode(model,input_data,out_length=None,starts=None):
    #保证随机旋转编码起始位置的方案可以正常工作
    global use_random
    use_random = True
    if model.model_type == "translator":
        if starts is None:
            answer = torch.ones(input_data[0].size(0), 1, dtype=torch.int32).to(input_data[0].device)
        else:
            answer = starts
        emb_querys = [embedding(query) for embedding,query in zip(model.encoder_embeddings,input_data)]
        memory = model.encoder_group(emb_querys,input_data[0]!=0)
        assert out_length is not None, "out_length is None"
        for _ in range(out_length):
            emb_answer = model.decoder_embedding(answer)
            out = model.decoder(emb_answer,answer!=0,memory)
            prob_dist = model.projector(out[:,-1:,:])
            next_token = torch.max(prob_dist, dim = -1)[1]
            answer = torch.cat([answer,next_token], dim=-1)
        return answer[:,1:]
    if model.model_type == "classifier":
        prob_dist = model.forward(input_data, input_data[0]!=0)
        return torch.max(prob_dist, dim = -1)[1].data[:,0]
    if model.model_type == "vit":
        prob_dist = model.forward(
            input_data,
            torch.from_numpy(np.ones((input_data[0].size(0),196),dtype='bool')).to(input_data[0].device)
        )
        return torch.max(prob_dist, dim = -1)[1].data[:,0]
    if model.model_type == "generator":
        sequence = input_data[0]
        assert out_length is not None, "out_length is None"
        for _ in range(out_length):
            emb_query = model.encoder_embeddings[0](sequence)
            prob_dist = model.projector(model.encoder_group([emb_query],q_mask)[:,-1:,:])
            next_token = torch.max(prob_dist, dim = 1)[1].data[0]
            sequence = torch.cat([sequence, torch.ones(1, 1, dtype=torch.int32).fill_(next_token).to(input_data[0].device)], dim=-1)
        return sequence
            
# random_decode