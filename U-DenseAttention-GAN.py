import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()
        
    def forward(self, x):
        return (torch.tanh(block8) + 1) / 2
      
      
#class BlockConv(nn.Module):
#   def __init__(self, channels):
        super(BlockConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
      
#   def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.relu(feature)
        feature = self.conv2(feature)
        return feature


class InBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(InBlock,self).__init__()
        self.in = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self,x):
        x = self.in(x)
        return x
    
class DownBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DownBlock,self).__init__()
        self.dn = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self,x):
        x = self.dn(x)
        return x
    
class BottomBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(BottomBlock,self).__init__()
        self.bot = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.bot(x)
        return x
  

class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            # resampler ???
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
    
class UpBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpBlock,self).__init__()
        self.up = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self,x):
        x = self.dn(x)
        return x

######################
Img

1)
Cn,BN,ReLU
Cn
[skp cntion1]
MaxP

2)
Cn,BN,ReLU
Cn
[skp cntion2]
MaxP

3)
Cn,BN,ReLU
Cn
[skp cntion3]
MaxP

4)
Cn,BN,ReLU
Cn
[skp cntion4]
MaxP

5)
Cn,BN,ReLU

6)
gating_Signal4
UpSmp
Cn,BN,ReLU
Cn

7)
gating_Signal3
UpSmp
Cn,BN,ReLU
Cn

8)
gating_Signal2
UpSmp
Cn,BN,ReLU
Cn

9)
gating_Signal1
UpSmp
Cn,BN,ReLU
Cn

###################

Cn,BN,ReLU
Cn
Cn,BN,ReLU
Cn
Cn,BN,ReLU
Cn
Cn,BN,ReLU
Cn
Cn,BN,ReLU
Cn
Cn,BN,ReLU
Cn
Dense
Sigmoid

