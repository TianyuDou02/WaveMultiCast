import torch
from torch import nn
from math import log
class ChannelAttention(nn.Module):
  def __init__(self,channel,gamma=2,b=1):
    super().__init__()
    t = int(abs((log(channel,2)+b)/gamma))
    k = t if t%2 else t+1
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv1d(1,1,kernel_size=k,padding=k//2,bias=False)
    self.sigmoid = nn.Sigmoid()
  def forward(self,x):
    y = self.avg_pool(x)
    y = self.conv(y.squeeze(-1).transpose(-1,-2))
    y = y.transpose(-1,-2).unsqueeze(-1)
    y = self.sigmoid(y)
    return x*y.expand_as(x)
