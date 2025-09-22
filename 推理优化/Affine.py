import torch
import torch.nn as nn
#仿射变换
class Affine(nn.Module):
    def __init__(self,value,grad_factor=100):
        super(Affine,self).__init__()
        self.value = nn.Parameter(torch.ones(1)*value/grad_factor)
        self.bias = nn.Parameter(torch.zeros(1))
        self.grad_factor=grad_factor
        
    def forward(self, x):
        return (x*self.value+self.bias)*self.grad_factor