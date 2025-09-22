import torch
import torch.nn as nn
from Affine import Affine
#投射器
class Projector(nn.Module):
    def __init__(self,embedding_dim,vocab_out_size,enable_affine):
        super(Projector,self).__init__()
        self.enable_affine = enable_affine
        self.project = nn.Linear(embedding_dim,vocab_out_size,bias=False)
        if enable_affine:
            self.affine = Affine(1.0)
            
    def forward(self, x):
        if self.enable_affine:
            ori_dist = self.affine(self.project(x))
        else:
            ori_dist = self.project(x)
        return ori_dist
#生成器
class Generator(nn.Module):
    def __init__(self,embedding,encoder,projector):
        super(Generator, self).__init__()
        self.model_type         = "generator"
        self.embedding          = embedding
        self.encoder            = encoder
        self.projector          = projector

    def forward(self,inputs,q_mask):
        query = self.embedding(inputs)
        out = self.encoder(query,q_mask)
        return self.projector(out)
