# 生成式语言模型全流程实验
开源从中文分词器设计到推理优化的大语言模型全流程实验 \
[清洗后的数据](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT) \
[预训练得到的模型参数](https://huggingface.co/mdokl/Jerry-v0.01-0.18B) \
[在线体验(modelscope)](https://modelscope.cn/studios/xizhang123/zh_0.18B_LLM) \
[在线体验(huggingface)](https://huggingface.co/spaces/mdokl/zh_0.18B_LLM) \
每个文件夹中有对应的说明文件，其中更加详细的描述 \
原本这个项目是想作为一个LLM入门教程，但经历有限，自身水平更加有限，最终代码趋于屎山，时间久了，关键点自身都有遗忘，因此也适合作为教程。 \
由于想要真实的看到智能涌现，以及流程的完整，原始数据集选择了WuDaoCorpora2.0，这使得数据清洗，词表创建，模型训练的时间成本极高 \
不算花费的人力成本（这是最主要的），租用服务器会花费400元左右 \
... \
依赖：numpy,pytorch,matplotlib,ahocorasick \
注意：AC自动机要BYTES=1的版本！\
AHOCORASICK_BYTES=1 pip install git+https://github.com/WojciechMula/pyahocorasick.git \
训练条件：租用单卡4090一周 \
数据集：悟道200G公开数据集，去重筛选得到60G \
分词器：借鉴于[bytepiece](https://spaces.ac.cn/archives/9752)，基于AC自动机与最大概率路径算法 \
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
