import numpy as np

#加载词表
with open('vocab_b_65544.txt','r',encoding='utf-8') as f:
# with open('vocab_tiny_random.txt','r',encoding='utf-8') as f:
    words_count = dict()
    for word in f:
        if word[0] != '\t':
            k,v = word.split('\t')
            words_count[k] = int(v[:-1])
        
#补充缺失词，但尽量不要改变词频
if '.' in words_count:
    words_count[','] = words_count['.']
    words_count['\r'] = 1
    words_count['\n'] = 1
    words_count['\t'] = 1
    
#计算每个片段长度的单词总数
N = 7
count_sum = [0 for _ in range(N)]
for k,v in words_count.items():
    count_sum[len(k)-1] += v

#创建AC自动机
import ahocorasick as ah
aca= ah.Automaton()
for k,v in words_count.items():
    aca.add_word(k.encode(),(len(k.encode()),np.log(v/count_sum[len(k)-1])))
aca.make_automaton()

#单词与整数互转字典
words = [k for k in words_count]
words.sort()
word2idx = {k.encode():i+1 for i,k in enumerate(words)}
idx2word = {i:k for k,i in word2idx.items()}
vocab_size = len(word2idx)

#分词器函数
def tokenizer(text,alpha=1.0):
    encode_text = text.encode()
    #路径，记录起始位置和分值
    LOT = len(encode_text)
    BOW = 0 #表示最佳词的起始位置
    VOW = 1 #表示最佳路径的累积值
    VOID = 5 #表示没有记录
    routes = [(i,VOID) for i in range(LOT)] + [(-1,0.0)]
    tokens = []  #保存分词结果
    #遍历所有匹配成功的词
    # low:len_of_word
    # vow:value_of_word
    for eow, (low,vow) in aca.iter(encode_text):
        #匹配词起点序号 = 匹配词终点序号 -（匹配词长度-1）
        bow = eow - low + 1
        #得分是负数，但负的程度越小约好，
        #数值为起始位置的得分 + 当前词的分数
        #起始位置无记录就往前找
        i = 0
        while routes[bow - 1 - i][VOW] == VOID:
            i += 1
        v = routes[bow - 1 -i][VOW] + vow
        # 超过5.0直接使用确定算法
        if alpha >= 5.0:
            #更短的路径或第一个到达，更新
            if v > routes[eow][VOW] and i == 0 or routes[eow][VOW] == VOID:
                routes[eow] = bow,v #记录起始位置以及累积值
        else:
            # 随机算法
            if routes[eow][VOW] == VOID:
                base = v
                temp = 1.0
                denominator = 1.0
                routes[eow] = bow,v
            else:
                temp = np.exp(alpha * (v - base))
                denominator += temp
                if np.random.rand() < temp/denominator and i == 0:
                    routes[eow] = bow,v #记录起始位置以及累积值
    #从后往前查找分割点
    eow = LOT - 1
    while encode_text:
        bow = routes[eow][BOW] #找到最佳词的起始位置
        tokens.append(encode_text[bow:eow+1]) #记录该词语
        encode_text,eow = encode_text[:bow],bow - 1 #继续分上一个词
    #从后往前找，需要反序得到正序的分词结果
    return [word2idx[w] if w in word2idx else -ord(w) for w in tokens[::-1]]

def token2str(tokens,split=''):
    return b''.join([(int(-token)).to_bytes(1,'big') if token < 0 else idx2word[token] + split.encode() for token in tokens]).decode(errors="ignore")
    

