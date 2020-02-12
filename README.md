# U-net
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
1)
Cn,BN,ReLU
Cn
2)
Cn,BN,ReLU
Cn
3)
Cn,BN,ReLU
Cn
4)
Cn,BN,ReLU
Cn
5)
Cn,BN,ReLU
Cn
6)
Cn,BN,ReLU
Cn
7)
Dense
Sigmoid
################
[Attention U-Net:Learning Where to Look for the Pancreas]
[U-GAN: Generative Adversarial Networks with UNet for Retinal Vessel Segmentation]
