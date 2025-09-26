# 生成式语言模型全流程实验
## 相关链接
[清洗后的数据](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT) \
[预训练得到的模型参数,Base版本](https://huggingface.co/mdokl/Jerry-v0.01-0.18B) \
[在线体验(modelscope,SFT版本)](https://modelscope.cn/studios/xizhang123/zh_0.18B_LLM) \
[在线体验(huggingface,SFT版本)](https://huggingface.co/spaces/mdokl/zh_0.18B_LLM) \
每个文件夹中有对应的说明文件，其中有相对更加详细的描述 
## 指令微调版本测试截图
<img width="1828" height="905" alt="image" src="https://github.com/user-attachments/assets/32e1a70f-ce29-48a3-bcb0-b05af9d1ba03" />

<img width="1506" height="384" alt="e12404928b0ff11bfaf7a71c8e19dd2" src="https://github.com/user-attachments/assets/e046d7c4-2206-46a9-9818-86eb1f115e7c" />

## 简要介绍
这是一个完整的LLM项目，从数据清洗和分词器设计开始，直到指令微调结束。 \
现在已经完成指令微调，数据选自[匠数科技大模型sft数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)。 \
未来，在主线方面还会有强化学习以及人类对齐相关的内容。 \
在支线方面，还会尝试优化、重构当前的模块，添加足够易读的注释和文档。 \
在更遥远的未来，或许还会涉及到CUDA编程，甚至重写深度学习框架，使项目更加完整。 \
注意，不同文件夹中的同名文件，内容可能有细微差异，比如EL-Attention只用于推理加速，训练时是没有的。 \
项目才刚刚开始，许多模块还属于屎山代码，需要不断优化。
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
inv任务：输入序列反向输出，用于验证模型全局位置信息捕捉能力 \
实验结果，在position_information_type = "mask" 的条件下，模型能够轻松解决add和inv任务 \
只需要16，32，64，三个序列长度上进行训练，模型就能够**准确**将128甚至更长的序列反向， 具有非常良好的长度外推能力 \
对于序列反向任务，使用正余弦位置编码、旋转位置编码只在训练过的长度上表现良好，而掩码位置信息几乎在所有长度上都可工作 \
<img width="551" height="420" alt="image" src="https://github.com/user-attachments/assets/407b4e12-b632-4480-b505-7ea65248ff88" /> \
但如果添加一个标记，让模型同时学习add和reverse任务，测试时根据标记切换，则reverse任务难以完成，说明方案具有一定的局限性 \
enable_affine = True 可以在预实验中让模型**极其迅速的收敛**，并可能增强模型能力。 \
经过测试，enable_affine = True的ViT模型在Cifar10上的loss会低于正常模型，测试集准确率更高 \
<img width="570" height="419" alt="image" src="https://github.com/user-attachments/assets/da572df7-014d-47e7-af7c-442dae392ca1" /> \
<img width="545" height="410" alt="image" src="https://github.com/user-attachments/assets/6e9c9db6-a737-428f-8c0e-0ae90d1b437a" /> \
预实验是对这些特性可扩展能力的测试，从结果上看，部分特性是有效的
## 词表创建与数据清洗
1、初始化词表，记录所有长度为1～7的片段，可以只近似保留词频在1/1000000的片段，否则初始词表的体积以及处理过程消耗的内存非常巨大。 \
2、词表迭代，加载初始词表，用最大概率路径算法对整个语料分词，记录出现过的词汇。（这里用AC自动机进行多模匹配，分词速率可以在1Mtokens/s，可以并行加速。） \
3、不断迭代，直到词表体积稳定。 \
4、在词表中查找连续的长片段，这些就是广告词，用于清洗广告。 \
5、手动缩减词表大小，再分词迭代，直到缩减到合适的大小。\
关于数据清洗的更多细节见[清洗后的数据](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT) 
## 预训练loss变化
使用Muon优化器，但前期配置错误，等价于使用Adam优化器，50000时及时修复，没有再出现过损失尖刺spike。\
训练片段长度不断增加，每次增加都可以看到loss的突然下降，最终片段长度为1024token，大约3000字/样本。\
<img width="551" height="413" alt="image" src="https://github.com/user-attachments/assets/753726a6-8f2b-4e6f-85b2-845fa4fc4a3b" />
