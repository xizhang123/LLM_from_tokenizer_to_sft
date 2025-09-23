# 生成式语言模型全流程实验
## 相关链接
[清洗后的数据](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT) \
[预训练得到的模型参数](https://huggingface.co/mdokl/Jerry-v0.01-0.18B) \
[在线体验(modelscope)](https://modelscope.cn/studios/xizhang123/zh_0.18B_LLM) \
[在线体验(huggingface)](https://huggingface.co/spaces/mdokl/zh_0.18B_LLM) \
每个文件夹中有对应的说明文件，其中有相对更加详细的描述 
## 简要介绍
这是一个完整的LLM项目，从数据清洗和分词器设计开始，直到预训练结束。 \
未来，在主线方面还会有强化学习以及人类对齐相关的内容。 \
在支线方面，还会尝试优化、重构当前的模块，添加足够易读的注释和文档。 \
在更遥远的未来，或许还会涉及到CUDA编程，甚至重写深度学习框架，使项目更加完整。 \
但在现在，项目才刚刚开始，许多模块还属于屎山代码，一切会慢慢变好的。 
## 一些额外说明
原始数据集选择了WuDaoCorpora2.0，数据清洗的流程还有待优化，当前清洗策略大约会花费一周时间。 \
如果使用清洗后的数据，租用服务器进行预训练，会花费4090 24G 一周。 \
训练过程使用的是单精度float32，并且暂时无法在半精度下保证不崩溃，因此这也是一个后续优化的目标。 \
当前训练时GPU功率可以接近100%，主要瓶颈是算力而非通信，租用A100或48G魔改版4090都只会增加成本，不会提高速度。 \
没有尝试过多卡训练，训练过程支持随时查看状态以及手动操作，但用户不友好。 \
[便宜的算力平台（智星云）链接](https://www.ai-galaxy.cn/)。 
## 特别注意
依赖：numpy,pytorch,matplotlib,ahocorasick \
由于分词器借鉴自[bytepiece](https://spaces.ac.cn/archives/9752)，所以依赖ahocorasick库。 \
在安装时请使用如下命令，否则切分粒度不是字节： \
AHOCORASICK_BYTES=1 pip install git+https://github.com/WojciechMula/pyahocorasick.git 
## 训练时模型参数设置
通用参数： \
vocab_size = 65544 + 1 + 255 \
embedding_dim = 768 \
key_dim = 128 \
head_number = 12 \
feed_forward_dim = 1536 \
deep = 12 \
enable_talking_head = True \
enable_layer_norm = True \
dropout_rate = 0.1 \
特殊参数： \
position_information_type = "mask" (可选"sinusoidal","rotary","learned") \
enable_affine = True (预实验中可以显著加速收敛，收敛速度块5倍以上，甚至能完成原本无法解决的任务) \
self_sttention_block_size = 0 (不通过分块的方式节省资源) 
## 预实验
预实验是在更古老的代码上进行的，当时为编码解码结构。 \
这一部分用于说明一些奇怪设置的由来，会在近期补充。 \
copy任务：正确性验证 \
add任务：相邻数字相加，用于验证模型相对位置信息的捕捉能力 \
reverse任务：输入序列反向输出，用于验证模型全局位置信息捕捉能力 \
实验结果，在position_information_type = "mask" 的条件下，模型能够轻松解决add和reverse任务 \
只需要16，32，64，三个序列长度上进行训练，模型就能够**准确**将128甚至更长的序列反向， 具有非常良好的长度外推能力 \
但如果添加一个标记，让模型同时学习add和reverse任务，测试时根据标记切换，则reverse任务难以完成，说明方案具有一定的局限性 \
enable_affine = True 可以在预实验中让模型**极其迅速迅速的收敛** \
经过测试，enable_affine = True的ViT模型在Cifar10上的loss会低于正常模型 \
实验是对这些特性可扩展能力的测试，从结果上看，部分特性是有效的 \
**在线体验中的是未经过微调的base模型，不会遵循指令，但具有基本的上下文捕捉能力，有进行指令微调的潜力** \
上下文捕捉测试参数设置： \
Repeat Penalty Decay Rate = **1.0** \
Repeat Penalty = **0.0** \
Temperature = 0.0001 \
上下文捕捉测试问题示例：\
**A = pzjendglaomwftqfduau785e39，这里添加一些扰动信息，今天天气真好，万里无云，乌云密布。你是一个语言模型。扰动信息结束，A =**\
就能看到模型准确的输出A对应的字符串。 \
<img width="1482" height="349" alt="image" src="https://github.com/user-attachments/assets/4198a347-44bb-46da-a2c3-8aceff33b6ff" /> 


## 以下的内容暂未整理，随便看看
词表：
- 从n-gram片段开始，经过12轮全自动迭代得到， 
- 65544词，包含常用词语，生僻字支持字节拆分， 
- 字母数字按照字符拆分 
  
模型：Decoder-Only，0.18B \
位置编码：借鉴RoPE，与Alibi极其相似（真无限外推，能完成任意长度而的序列反向任务，具体原因有待分析） \
优化器：前半段Adam，后期Muon \
技巧：梯度累加模拟大批次，关键变量缓存加速 \
技巧：序列长度从128->256->512->1024变化，前期加速每后期长上下文 \
训练性能：后期seq=1024时吞吐量17990.7 tokens/s，前期没有记录，但处理完60G数据总共用时一周。 \
推理优化：EL-Attention \
微调数据：COIG-CQIA-full（数据量不太够，这里没有整理微调相关的部分）

训练loss（每次增加片段长度，loss会有突然降低，这也证明了位置编码的长度外对能力良好）: \
需要说明的是，在训练长度为128时，模型具有良好的文采，但后期，随着序列长度增加，就成为了普通的语言模型，因此小模型的最优训练片段长度值得探索。 \
<img width="551" height="413" alt="image" src="https://github.com/user-attachments/assets/753726a6-8f2b-4e6f-85b2-845fa4fc4a3b" /> \
由于中间阶段的模型权重丢失了，这里只保留两段输出结果（没及时保存算是小小的遗憾，但最终的模型还是更加稳定通用的）
```
一去二三里,是间隔帘的灰烟,无论槐花树下的那个寂寂的空笑,是间离不开的恋曲,或是重逢的企盼,甚至是一种悠然的心情。踏上被人们称作"小家碧玉"的石板路,一路绵延不绝地在神庙的一边
```
```
相见时难别亦难,只因前世无缘却相聚,培训归来不足依依惜别。岁月如烟,刻画古今对家乡,对故乡的留恋,人世相逢却只能相拥而泣,是难以理解。 往事千山,相映成趣,相诉相随,泪流迭下,失望的无奈开始显露,感情又失败了再退缩,
```
训练学习率调整： \
<img width="590" height="413" alt="image" src="https://github.com/user-attachments/assets/9eacd255-4d2d-4760-996e-99f1e392e355" /> \
训练时测试（允许暂停训练随时查看效果，此时没有重复惩罚）： \
<img width="1794" height="434" alt="image" src="https://github.com/user-attachments/assets/3397efb2-69ae-4448-89c5-e5a7a90865df" /> \
推理优化（3070 mobile）： \
<img width="1711" height="779" alt="image" src="https://github.com/user-attachments/assets/f8ec8d15-cc47-4c1c-ae4e-9266a09246e2" /> \
<img width="1067" height="599" alt="image" src="https://github.com/user-attachments/assets/3057122e-eadb-414a-bdc3-2d806cd24161" />
