# 生成式语言模型全流程实验
开源从中文分词器设计到推理优化的大语言模型全流程实验 \
依赖：pytorch,ahocorasick \
注意：AC自动机要BYTES=1的版本！\
AHOCORASICK_BYTES=1 pip install git+https://github.com/WojciechMula/pyahocorasick.git \
训练条件：租用单卡4090一周 \
数据集：悟道200G公开数据集，去重筛选得到60G \
分词器：借鉴于bytepiece，基于AC自动机与最大概率路径算法 \
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
微调数据：COIG-CQIA-full \
...省略细节后续补充， \
这是第二次重构，依然还有许多缺陷，但由于研究方向与大模型关系不大，不能继续研究了，开源纪念这段探索大模型原理经历。 \

训练loss: \
<img width="551" height="413" alt="image" src="https://github.com/user-attachments/assets/753726a6-8f2b-4e6f-85b2-845fa4fc4a3b" />
训练学习率调整： \
<img width="590" height="413" alt="image" src="https://github.com/user-attachments/assets/9eacd255-4d2d-4760-996e-99f1e392e355" />
训练时测试（允许暂停训练随时查看效果，此时没有重复惩罚）： \
<img width="1794" height="434" alt="image" src="https://github.com/user-attachments/assets/3397efb2-69ae-4448-89c5-e5a7a90865df" />
推理优化（3070 mobile）： \
<img width="1711" height="779" alt="image" src="https://github.com/user-attachments/assets/f8ec8d15-cc47-4c1c-ae4e-9266a09246e2" />
<img width="1067" height="599" alt="image" src="https://github.com/user-attachments/assets/3057122e-eadb-414a-bdc3-2d806cd24161" />
现在专注于大模型使用，而非训练[狗头]
