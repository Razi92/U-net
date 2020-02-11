import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()
        
    def forward(self, x):
        return (torch.tanh(block8) + 1) / 2
      
      
class ResidualBlock(nn.Module):
    def __init__(self, channels):
      
    def forward(self, x):
        return (x + residual)
  
  
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
      
    def forward(self, x):
        return x

######################
Img

1)
Cn,BN,ReLU
Cn
[skp cntion1]
MaxP_from_DenseBlock

2)
Cn,BN,ReLU
Cn
[skp cntion2]
MaxP_from_DenseBlock

3)
Cn,BN,ReLU
Cn
[skp cntion3]
MaxP_from_DenseBlock

4)
Cn,BN,ReLU
Cn
[skp cntion4]
MaxP_from_DenseBlock

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


