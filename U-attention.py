# U net with attention gates
import math
import torch
from torch import nn


class U_Att(nn.Module):
	
    def __init__(self, input_channels, output_channels):
        # upsample_block_num = int(math.log(scale_factor, 2))
        # ??? Is this necessary?
        super(U_Att, self).__init__()
        self.ch_in = input_channels
        self.ch_out = output_channels
	
        self.block1 = InBlock(ch_in, ch_out)
        self.block2 = DownBlock(???)
        self.block3 = DownBlock(???)
        self.block4 = DownBlock(???)
        self.block5 = BottomBlock(???)
        self.block6 = AttentionBlock(???, input1, input2)
        self.block7 = UpBlock(???, input1, input2)
        self.block8 = AttentionBlock(???, input1, input2)
        self.block9 = UpBlock(???, input1, input2)
        self.block10 = AttentionBlock(???, input1, input2)
        self.block11 = UpBlock(???, input1, input2)
        self.block12 = AttentionBlock(???, input1, input2)
        self.block13 = UpBlock(???, input1, input2)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.block6(out4, out5)
        out7 = self.block7(out5, out6)
	out8 = self.block8(out3, out7)
        out9 = self.block9(out7, out8)
        out10 = self.block10(out2, out9)
        out11 = self.block11(out9, out10)
        out12 = self.block12(out1, out11)
        out13 = self.block13(out11, out12)

        return (torch.tanh(out13) + 1) / 2
      
      
    def __init__(self, input_channels, output_channels, feature_channels=32,
                 kernel_size_encoding=3, kernel_size_decoding=3, num_conv_layers=1,
                 leaky_slope=0.2, use_batch_norm=True, use_dropout=False, dropout_rate=0.2):


class InBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(InBlock,self).__init__()
        self.in = nn.Sequential(
		nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		nn.BatchNorm2d(ch_out),
		nn.ReLU(inplace=True),
		nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		# ??? is there a problem when the second Conv has the same in and out chanel size?
        )
    def forward(self, x):
        x = self.in(x)
        return x
    
	
class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownBlock,self).__init__()
        self.dn = nn.Sequential(
		nn.MaxPool2d(kernel_size=2, stride=2)
		nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		nn.BatchNorm2d(ch_out),
		nn.ReLU(inplace=True),
		nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		# ??? is there a problem when the second Conv has the same in and out chanel size?
        )
    def forward(self, x): 
        x = self.dn(x)
        return x
    
	
class BottomBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BottomBlock,self).__init__()
        self.bot = nn.Sequential(
		nn.MaxPool2d(kernel_size=2, stride=2)
		nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		nn.BatchNorm2d(ch_out),
		nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.bot(x)
        return x
  

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock,self).__init__()
        self.W_g = nn.Sequential(
            	nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            	nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            	nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
		nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
           	nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
           	nn.BatchNorm2d(1),
           	nn.Sigmoid()
		# ??? I don't get the dimentions of Conv2D
           	# resampler ???
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
    
	
class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpBlock,self).__init__()
	self.upsampling = nn.Upsample(scale_factor=2)
        self.up = nn.Sequential(
		nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		nn.BatchNorm2d(ch_out),
		nn.ReLU(inplace=True),
		nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
    def forward(self, x, y):
	feature1 = self.upsampling(x)
	# concatination of simple add ???
        feature2 = self.up(feature1+y)
        return feature2



'''class BlockConv(nn.Module):
   def __init__(self, channels):
        super(BlockConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
      
   def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.relu(feature)
        feature = self.conv2(feature)
        return feature'''
###
'''class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x'''
