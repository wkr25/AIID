import torch
from torch import nn
import torch.nn.functional as F

class Convolution(nn.Module):
    def __init__(self,channel_in,channel_out):
        super().__init__()
        self.Conv=nn.Sequential(
            nn.Conv2d(channel_in,channel_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self,x):
        return self.Conv(x)
#卷积层（非常标准的卷积->标准化->非线性层->池化）

class Detector(nn.Module):
    def __init__(self,embedding_dim=128,num_class=2):
        super().__init__()
        self.feature=nn.Sequential(
            Convolution(3,32),
            Convolution(32,64),
            Convolution(64,128),
            Convolution(128,256),
        )
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.flatten=nn.Flatten()
        self.embedding_layer=nn.Linear(256,embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_class)
    def forward(self,x):
        x=self.feature(x)
        x=self.pool(x)
        x=self.flatten(x)
        tmp=self.embedding_layer(x)
        logits=self.classifier(tmp)
        embedding=F.normalize(tmp,dim=1)
        return logits,embedding
#核心网络（四个卷积层->压缩层->线性层），这里的思路是通过几层网络将图像信息压缩为向量，并转为两个结果：1.“AI率”判别 2.向量（可用于进一步的contrastive loss可能性探索）