import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import threading
import copy
class Batch:
    def __init__(self,input_sequences):
        self.data_type = "generator"
        self.query = input_sequences[...,:-1]
        self.label = input_sequences[...,1:]
        self.q_mask = self.query != 0
        self.ntokens = float((self.label != 0).sum())

#交叉熵损失，“0”填充特殊处理
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # 使用KL散度损失函数（接受对数概率分布x和概率分布y,并不是简单的KL散度计算）
        self.criterion = nn.KLDivLoss(reduction='sum')
        
    def forward(self, model_output_dist, target_sequence):
        #根据模型输出的分布与标签的分布计算交叉熵损失
        #为目标分布分配和模型输出形状、类型一样的空间，默认不追踪梯度，写明更清晰
        true_dist = torch.zeros_like(model_output_dist,requires_grad=False)
        #使用置信度填充目标词的位置（true_dist是词表那么长的概率分布）
        #目标序列升维，target_sequence:[batch*len]->[batch*len,1]
        #true_dist:[batch*len,vocab]
        #在vocab的维度上用标签值当作索引，找到对应元素，填充1.0
        true_dist.scatter_(1, target_sequence.data.unsqueeze(1), 1.0)
        #将填充位置概率设为0
        true_dist[:,0] = 0.0
        #计算模型输出分布与平目标序列标签平滑后的分布之间的交叉熵
        #model_output_dist是对数概率分布,应由F.log_softmax(self.project(x),dim=-1)产生
        #但实际上为了压缩softmax的值域已达到自动丢弃异常值的效果，在Generator.Projector进行了特殊实现
        return self.criterion(model_output_dist, true_dist)

class SimpleAdamOptimizer:
    "简单的自适应矩估计优化器"
    def __init__(self, params, betas, eps, weight_decay):
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.param_groups = []
        for p in params:
            self.param_groups.append({
                'params': p, 
                'lr'    : 0.0, 
                'm'     : torch.zeros_like(p).detach(), 
                'v'     : torch.zeros_like(p).detach()
            })

    def step(self):
        self.t += 1
        for group in self.param_groups:
            # 获取梯度
            grad = group['params'].grad
            if grad is None:
                continue
            grad[grad!=grad] = 0.0
            with torch.no_grad():
                # 历史衰减
                group['m'].mul_(self.beta1).add_(grad, alpha = 1 - self.beta1)
                group['v'].mul_(self.beta2).addcmul_(grad, grad, value = 1 - self.beta2)
                # 偏差纠正
                m_hat = group['m'] / (1 - self.beta1 ** self.t)
                v_hat = group['v'] / (1 - self.beta2 ** self.t)
                # 参数更新
                group['params'].sub_(group['lr'] * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * group['params']))

    def zero_grad(self):
        for group in self.param_groups:
            if group['params'].grad is not None:
                group['params'].grad.detach_()
                group['params'].grad.zero_()
# 显存爆炸，借来一用
# def Msign(M):
#     U,S,Vh = torch.linalg.svd(M,full_matrices=True)
#     return U[...,:S.size(-1)]@Vh[...,:S.size(-1)]
    
def Msign(G, steps=20,eps=1e-7):
    # a,b,c = (3.4445,-4.7750,2.0315)
    a,b,c = (15/8,-5/4,3/8)
    X = G/(G.norm() + eps)
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1,-2)
    for _ in range(steps):
        A = X @ X.transpose(-1,-2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1,-2)
    return X

class SimpleMoOptimizer:
    "简单的动量正交化优化器"
    #如果是二维矩阵，就用动量正交化
    #如果是其他类型，就继续用自适应据估计
    def __init__(self, params, betas, eps, weight_decay, get_info):
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.param_groups = []
        for i,p in enumerate(params):
            self.param_groups.append({
                'params': p, 
                'lr'    : 0.0, 
                'm'     : torch.zeros_like(p).detach(), 
                'v'     : torch.zeros_like(p).detach() if get_info(i,p) == 'vector' else None,
                'info'  : get_info(i,p)
            })

    def step(self):
        self.t += 1
        for group in self.param_groups:
            # 获取梯度
            grad = group['params'].grad
            if grad is None:
                continue
            grad[grad!=grad] = 0.0
            with torch.no_grad():
                if group['info'] == 'vector':
                    # 历史衰减
                    group['m'].mul_(self.beta1).add_(grad, alpha = 1 - self.beta1)
                    group['v'].mul_(self.beta2).addcmul_(grad, grad, value = 1 - self.beta2)
                    # 偏差纠正
                    m_hat = group['m'] / (1 - self.beta1 ** self.t)
                    v_hat = group['v'] / (1 - self.beta2 ** self.t)
                    # 参数更新
                    group['params'].sub_(group['lr'] * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * group['params']))
                elif group['info'] == 'matrix':
                    group['m'].mul_(self.beta1).add_(grad, alpha = 1 - self.beta1)
                    scale = 0.2 * np.sqrt(max(group['m'].size(-1),group['m'].size(-2)))
                    group['params'].sub_(group['lr'] * (scale * Msign(group['m']) + self.weight_decay * group['params']))
                else:
                    group['m'].mul_(self.beta1).add_(grad, alpha = 1 - self.beta1)
                    head_number,key_dim,embedding_dim = group['info']
                    scale = 0.2 * np.sqrt(max(key_dim,embedding_dim))
                    g_split_by_head = grad.view(head_number,key_dim,embedding_dim)
                    m_split_by_head = group['m'].view(head_number,key_dim,embedding_dim)
                    p_split_by_head = group['params'].view(head_number,key_dim,embedding_dim)
                    p_split_by_head.sub_(group['lr'] * (scale * Msign(m_split_by_head) + self.weight_decay * p_split_by_head))

    def zero_grad(self):
        for group in self.param_groups:
            if group['params'].grad is not None:
                group['params'].grad.detach_()
                group['params'].grad.zero_()

def get_lrate(start_step,total_step,lr_from,lr_to,transition,enable_wave):
    assert transition > 0 and transition % 2 == 0, "Need transition lt 0 and transition mod 2 eq 0."
    mid_transition = transition // 2
    half_lr_gap = (lr_to - lr_from)/2
    if total_step >= start_step + transition:
        ret = lr_to
    elif total_step < start_step + mid_transition:
        ret = lr_from + half_lr_gap * (total_step - start_step)**2 / mid_transition**2
    else:
        ret =  lr_to - half_lr_gap * (start_step + transition - total_step)**2 / mid_transition**2
    #最后的时候震荡，否则有危害
    if ret != lr_to or enable_wave == False:
        return ret
    else:
        x = (total_step - start_step) / mid_transition
        x += 0.1
        x -= int(x)
        x -= 0.1
        if x >= 0:
            x /= 0.9
        else:
            x /= 0.1
        x *= np.pi
        return ret * ((np.cos(x)*0.5+0.5)*0.9+0.1)

record = {
    "loss_line" : [],
    "lr_line" : []
}

class OptimizerWrapper:
    def __init__(self, optimizer, warm_up, lr, enable_wave = False):
        self.lr_from   = 0                 #初始学习率
        self.lr_to     = lr                #目标学习率
        self.warm_up   = warm_up           #预热步数
        self.start_step= 0                 #起始步数
        self.total_step= 0                 #总步数
        self.optimizer = optimizer         #优化器，用于执行梯度下降
        self.enable_wave = enable_wave     #学习率波动

    def update(self):
        global record
        #设置优化器中每个参数组的学习率并执行梯度下降
        lrate = self.lrate()
        record["lr_line"] += [lrate]
        for parameters in self.optimizer.param_groups:
            parameters['lr'] = lrate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def lrate(self):
        self.total_step += 1
        return get_lrate(
            self.start_step,
            self.total_step,
            self.lr_from,
            self.lr_to,
            self.warm_up,
            self.enable_wave)
        
    def set_lrate(self,lrate,transition):
        self.lr_from = self.lr_to
        self.lr_to = lrate
        self.warm_up = transition
        self.start_step = self.total_step

stop = False
pause = False
caculate_size_manual = 0

def TOGGLE():
    global pause
    pause = not pause
    print("pause:",pause)
    
def STOP():
    global stop
    stop = True

def SET_CACULATE_SIZE(caculate_size):
    global pause
    global caculate_size_manual
    assert pause == True ,"stop to set caculate_size."
    caculate_size_manual = caculate_size

def run_epoch(model,data_iter,caculate_size,loss_f,optimizer,epoch,use_amp):
    global stop
    global pause
    global record
    global caculate_size_manual
    for step, batch in enumerate(data_iter):
        if stop:
            break
        while pause:
            time.sleep(0.5)
        if caculate_size_manual != 0:
            caculate_size = caculate_size_manual
        total_loss = 0
        t_start = time.time()
        for i in range(0,batch.query.size(0),caculate_size):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model_output = model(batch.query[i:i+caculate_size], batch.q_mask[i:i+caculate_size])
                    loss = loss_f(torch.log_softmax(model_output,dim=-1).view(-1,model_output.size(-1)),
                        batch.label[i:i+caculate_size].reshape(-1))/ batch.ntokens
                    loss.backward()
                    total_loss += float(loss) * batch.ntokens
            else:
                model_output = model(batch.query[i:i+caculate_size], batch.q_mask[i:i+caculate_size])
                loss = loss_f(torch.log_softmax(model_output,dim=-1).view(-1,model_output.size(-1)),
                    batch.label[i:i+caculate_size].reshape(-1))/ batch.ntokens
                loss.backward()
                total_loss += float(loss) * batch.ntokens
        optimizer.update()
        mean_loss = total_loss/batch.ntokens
        record["loss_line"] += [mean_loss]
        t_end = time.time()
        print('epoch:',epoch,'\tstep:',step,'\tloss:',str(mean_loss)[:5],'\tspeed:',str(batch.ntokens/(t_end - t_start))[:7],'tokens/s',end = ' '*20)

            
#训练函数以服务模式运行，可以随时手动调整
def train(model,data_generator,batch_size,caculate_size,loss_f,optimizer,use_amp):
    global stop
    epoch = 0
    while(True):
        if stop:
            break
        run_epoch(model,data_generator(batch_size),caculate_size,loss_f,optimizer,epoch,use_amp)
        epoch += 1

#启动训练服务
def train_server_start(model,generator_batch_pair,caculate_size,loss_f,optimizer,use_amp = False):
    assert generator_batch_pair[1] % caculate_size == 0, "Need batch_size mod caculate_size eq 0."
    data_generator,batch_size = generator_batch_pair
    thread = threading.Thread(target=train,args=(model,data_generator,batch_size,caculate_size,loss_f,optimizer,use_amp))
    thread.start()
    

#贪婪解码
def greedy_decode(model,inputs,out_length):
    if model.model_type == "generator":
        for _ in range(out_length):
            query = model.embedding(inputs)
            prob_dist = model.projector(model.encoder(query,inputs==inputs)[:,-1:,:])
            next_token = torch.max(prob_dist, dim = -1)[1]
            inputs = torch.cat([inputs,next_token.to(inputs.device)], dim=-1)
        return inputs

#概率解码
def sampling_decode(model,inputs,out_length):
    if model.model_type == "generator":
        for _ in range(out_length):
            query = model.embedding(inputs)
            prob_dist = model.projector(model.encoder(query,inputs==inputs)[:,-1,:])
            next_token = torch.multinomial(F.softmax(prob_dist, dim = -1), num_samples = 1)
            inputs = torch.cat([inputs,next_token.to(inputs.device)], dim=-1)
        return inputs

#更可控的文本续写工具
def text_continue(model,inputs,out_length,repeat_penalty_value,temperature,decay=0.98):
    if model.model_type == "generator":
        repeat_penalty = None
        for _ in range(out_length):
            query = model.embedding(inputs)
            prob_dist = model.projector(model.encoder(query,inputs==inputs)[:,-1,:])
            if repeat_penalty is None:
                repeat_penalty = torch.zeros_like(prob_dist, device=inputs.device)
                for index in range(inputs.size(1)):
                    for line in range(inputs.size(0)):
                        repeat_penalty[line][inputs[line][index]] -= repeat_penalty_value
                    repeat_penalty *= decay
            else:
                repeat_penalty *= decay
            prob_dist += repeat_penalty
            next_token = torch.multinomial(F.softmax(prob_dist/temperature, dim = -1), num_samples = 1)
            inputs = torch.cat([inputs,next_token.to(inputs.device)], dim=-1)
            for i in range(next_token.size(0)):
                repeat_penalty[i][next_token[i]] -= repeat_penalty_value
        return inputs

#值函数，给基于蒙特卡洛树的续写用
def text_continue_value(model,inputs,out_length,repeat_penalty,repeat_penalty_value,temperature,decay):
    if model.model_type == "generator":
        ret = 0
        for _ in range(out_length):
            query = model.embedding(inputs)
            prob_dist = model.projector(model.encoder(query,inputs==inputs)[:,-1,:])
            prob_dist += repeat_penalty
            repeat_penalty *= decay
            prob_dist = F.softmax(prob_dist/temperature, dim = -1)
            next_token = torch.multinomial(prob_dist, num_samples = 1)
            inputs = torch.cat([inputs,next_token.to(inputs.device)], dim=-1)
            for i in range(next_token.size(0)):
                repeat_penalty[i][next_token[i]] -= repeat_penalty_value
                ret += prob_dist[i,next_token[i]]
        return ret

#基于蒙特卡洛树的续写
def RF_continue(model,inputs,out_length,repeat_penalty_value,temperature,try_n,acc_n,deep_n,decay=0.98):
    if model.model_type == "generator":
        repeat_penalty = None
        assert inputs.dim() == 1, "Need inputs.dim eq 1"
        inputs = inputs.repeat(try_n,1)
        for cur in range(out_length):
            query = model.embedding(inputs)
            prob_dist = model.projector(model.encoder(query,inputs==inputs)[:,-1,:])
            if repeat_penalty is None:
                repeat_penalty = torch.zeros_like(prob_dist, device=inputs.device)
                for index in range(inputs.size(1)):
                    for line in range(inputs.size(0)):
                        repeat_penalty[line][inputs[line][index]] -= repeat_penalty_value
                    repeat_penalty *= decay
            else:
                repeat_penalty *= decay
            prob_dist += repeat_penalty
            prob_dist = F.softmax(prob_dist/temperature, dim = -1)
            next_token = torch.multinomial(prob_dist, num_samples = 1)
            inputs = torch.cat([inputs,next_token.to(inputs.device)], dim=-1)
            values = []
            for i in range(next_token.size(0)):
                repeat_penalty[i][next_token[i]] -= repeat_penalty_value
                values += [prob_dist[i,next_token[i]]]
            max_v = 0.0
            max_i = 0
            cnt = 0
            for test_input,test_repeat_penalty,value in zip(inputs,repeat_penalty,values):
                test_input = test_input.repeat(acc_n,1)
                test_repeat_penalty = test_repeat_penalty.repeat(acc_n,1)
                value += float(text_continue_value(
                    model,test_input,deep_n,test_repeat_penalty,repeat_penalty_value,temperature,decay
                ))/(acc_n*deep_n)
                if value > max_v:
                    max_v = value
                    max_i = cnt
                cnt += 1
            if cur != out_length - 1:
                inputs = inputs[max_i].repeat(try_n,1)
                repeat_penalty = repeat_penalty[max_i].repeat(try_n,1)
            else:
                outputs = inputs[max_i]
        return outputs