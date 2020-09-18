# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from torch import nn
from .uanet_parts import *
#from /home/sanping/codes/saliency/HUAN/resnext import ResNeXt101


class HUANet(nn.Module):
    def __init__(self, n_classes=1):
        super(HUANet, self).__init__()
		
	resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
		
	self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )

        self.predict0_1 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict0_2 = nn.Conv2d(256, 1, kernel_size=1)
		

        self.inc1 = inconv(256, 64)
        self.down11 = down(64, 128)
        self.down12 = down(128, 256)
        self.down13 = down(256, 512)
        self.down14 = down(512, 512)

        self.down11_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.down12_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.down13_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.down14_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
		
	self.inc2 = inconv(512, 64)
        self.down21 = down(64+1, 128)
        self.down22 = down(128+1, 256)
        self.down23 = down(256+1, 512)
        self.down24 = down(512+1, 512)

        self.down21_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.down22_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.down23_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.down24_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
		
		
	self.inc3 = inconv(1024, 64)
        self.down31 = down(64+1, 128)
        self.down32 = down(128+1, 256)
        self.down33 = down(256+1, 512)
        self.down34 = down(512+1, 512)

        self.down31_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.down32_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.down33_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.down34_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
		
		
        self.inc4 = inconv(2048, 64)
        self.down41 = down(64+1, 128)
        self.down42 = down(128+1, 256)
        self.down43 = down(256+1, 512)
        self.down44 = down(512+1, 512)

        self.down41_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.down42_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.down43_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.down44_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
		

        self.up14 = up(512, 512, n_classes)
        self.up13 = up(512, 256, n_classes)
        self.up12 = up(256, 128, n_classes)
        self.up11 = up(128, 64, n_classes)

        self.up14_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.up13_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.up12_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.up11_heat = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
		
	self.up24 = up(512+1, 512, n_classes)
        self.up23 = up(512+1, 256, n_classes)
        self.up22 = up(256+1, 128, n_classes)
        self.up21 = up(128+1, 64, n_classes)

        self.up24_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.up23_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.up22_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.up21_heat = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
		
		
	self.up34 = up(512+1, 512, n_classes)
        self.up33 = up(512+1, 256, n_classes)
        self.up32 = up(256+1, 128, n_classes)
        self.up31 = up(128+1, 64, n_classes)

        self.up34_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.up33_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.up32_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.up31_heat = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
		
	self.up44 = up(512+1, 512, n_classes)
        self.up43 = up(512+1, 256, n_classes)
        self.up42 = up(256+1, 128, n_classes)
        self.up41 = up(128+1, 64, n_classes)

        self.up44_heat = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.up43_heat = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.up42_heat = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.up41_heat = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
	
        self.outc1 = outconv(64, n_classes)
		self.outc2 = outconv(64, n_classes)
		self.outc3 = outconv(64, n_classes)
		self.outc4 = outconv(64, n_classes)
		
	    for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
	    layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
		
	l0_size = layer0.size()[2:]
	reduce_low = self.reduce_low(torch.cat((layer0,
	                                        F.upsample(layer1, size=l0_size, mode='bilinear'),
						F.upsample(layer2, size=l0_size, mode='bilinear')), 1))
	reduce_high = self.reduce_high(torch.cat((layer3,
	                                         F.upsample(layer4, size=layer3.size()[2:], mode='bilinear')), 1)
        reduce_high = F.upsample(reduce_high, size=l0_size, mode='bilinear')
		
	pred0_1 = self.predict0_1(reduce_high)
	pred0_2 = self.predict0_2(reduce_low)
		
        x1 = self.inc1(F.upsample(layer1, l0_size, mode='bilinear'))
	x2 = self.inc2(F.upsample(layer2, l0_size, mode='bilinear'))
	x3 = self.inc3(F.upsample(layer3, l0_size, mode='bilinear'))
	x4 = self.inc4(F.upsample(layer4, l0_size, mode='bilinear'))
		
        x1 = self.down11(x1)
        down11_heat = self.down11_heat(x1)
	x2 = self.down21(torch.cat((x2, down11_heat), 1))
	down21_heat = self.down21_heat(x2)
	x3 = self.down31(torch.cat((x3, down21_heat), 1))
	down31_heat = self.down31_heat(x3)
	x4 = self.down41(torch.cat((x4, down31_heat), 1))
	down41_heat = self.down41_heat(x4)
		
        x1 = self.down12(x1)
        down12_heat = self.down12_heat(x1)
	x2 = self.down22(torch.cat((x2, down12_heat), 1))
	down22_heat = self.down21_heat(x2)
	x3 = self.down32(torch.cat((x3, down22_heat), 1))
	down32_heat = self.down32_heat(x3)
	x4 = self.down42(torch.cat((x4, down32_heat), 1))
	down42_heat = self.down42_heat(x4)
		
        x1 = self.down13(x1)
        down13_heat = self.down13_heat(x1)
	x2 = self.down23(torch.cat((x2, down13_heat), 1))
	down23_heat = self.down23_heat(x2)
	x3 = self.down33(torch.cat((x3, down23_heat), 1))
	down33_heat = self.down33_heat(x3)
	x4 = self.down43(torch.cat((x4, down33_heat), 1))
	down43_heat = self.down43_heat(x4)
		
        x1 = self.down14(x1)
        down14_heat = self.down14_heat(x1)
	x2 = self.down24(torch.cat((x2, down14_heat), 1))
	down24_heat = self.down24_heat(x2)
	x3 = self.down34(torch.cat((x3, down24_heat), 1))
	down34_heat = self.down34_heat(x3)
	x4 = self.down44(torch.cat((x4, down34_heat), 1))
	down44_heat = self.down44_heat(x4)
       
        x1 = self.up14(x1, 512, down14_heat)
        up14_heat = self.up14_heat(x1)
	x2 = self.up24(torch.cat((x2, up14_heat), 1), 512, down24_heat)
	up24_heat = self.up24_heat(x2)
	x3 = self.up34(torch.cat((x3, up24_heat), 1), 512, down34_heat)
	up34_heat = self.up34_heat(x3)
	x4 = self.up44(torch.cat((x4, up34_heat), 1), 512, down44_heat)
	up44_heat = self.up44_heat(x4)
		
        x1 = self.up13(x1, 256, down13_heat)
        up13_heat = self.up13_heat(x1)
	x2 = self.up23(torch.cat((x2, up13_heat), 1), 256, down23_heat)
	up23_heat = self.up23_heat(x2)
	x3 = self.up33(torch.cat((x3, up23_heat), 1), 256, down33_heat)
	up33_heat = self.up33_heat(x3)
	x4 = self.up43(torch.cat((x4, up33_heat), 1), 256, down34_heat)
	up43_heat = self.up43_heat(x4)
		
        x1 = self.up12(x1, 128, down12_heat)
        up12_heat = self.up12_heat(x1)
	x2 = self.up22(torch.cat((x2, up12_heat), 1), 128, down22_heat)
	up22_heat = self.up22_heat(x2)
	x3 = self.up32(torch.cat((x3, up22_heat), 1), 128, down32_heat)
	up32_heat = self.up32_heat(x3)
	x4 = self.up42(torch.cat((x4, up32_heat), 1), 128, down42_heat)
	up42_heat = self.up42_heat(x4)
		
        x1 = self.up11(x1, 64, down11_heat)
        up11_heat = self.up11_heat(x1)
	x2 = self.up21(torch.cat((x2, up11_heat), 1), 64, down21_heat)
	up21_heat = self.up21_heat(x2)
	x3 = self.up31(torch.cat((x3, up21_heat), 1), 64, down31_heat)
	up31_heat = self.up31_heat(x3)
	x4 = self.up41(torch.cat((x4, up31_heat), 1), 64, down41_heat)
        up41_heat = self.up41_heat(x4)
        
	x1 = self.outc1(x1)
	x2 = self.outc2(x2)
	x3 = self.outc3(x3)
	x4 = self.outc4(x4)
		
	pred0_1 = F.upsample(pred0_1, size=x.size()[2:], mode='bilinear')
	pred0_2 = F.upsample(pred0_2, size=x.size()[2:], mode='bilinear')
	d11 = F.upsample(down11_heat, size=x.size()[2:], mode='bilinear')
	d12 = F.upsample(down12_heat, size=x.size()[2:], mode='bilinear')
	d13 = F.upsample(down13_heat, size=x.size()[2:], mode='bilinear')
	d14 = F.upsample(down14_heat, size=x.size()[2:], mode='bilinear')
		
        d21 = F.upsample(down21_heat, size=x.size()[2:], mode='bilinear')
	d22 = F.upsample(down22_heat, size=x.size()[2:], mode='bilinear')
	d23 = F.upsample(down23_heat, size=x.size()[2:], mode='bilinear')
	d24 = F.upsample(down24_heat, size=x.size()[2:], mode='bilinear')
		
        d31 = F.upsample(down31_heat, size=x.size()[2:], mode='bilinear')
	d32 = F.upsample(down32_heat, size=x.size()[2:], mode='bilinear')
	d33 = F.upsample(down33_heat, size=x.size()[2:], mode='bilinear')
	d34 = F.upsample(down34_heat, size=x.size()[2:], mode='bilinear')
		
        d41 = F.upsample(down41_heat, size=x.size()[2:], mode='bilinear')
	d42 = F.upsample(down42_heat, size=x.size()[2:], mode='bilinear')
	d43 = F.upsample(down43_heat, size=x.size()[2:], mode='bilinear')
	d44 = F.upsample(down44_heat, size=x.size()[2:], mode='bilinear')
		
	u11 = F.upsample(up11_heat, size=x.size()[2:], mode='bilinear')
	u12 = F.upsample(up12_heat, size=x.size()[2:], mode='bilinear')
	u13 = F.upsample(up13_heat, size=x.size()[2:], mode='bilinear')
	u14 = F.upsample(up14_heat, size=x.size()[2:], mode='bilinear')
		
        u21 = F.upsample(up21_heat, size=x.size()[2:], mode='bilinear')
	u22 = F.upsample(up22_heat, size=x.size()[2:], mode='bilinear')
	u23 = F.upsample(up23_heat, size=x.size()[2:], mode='bilinear')
	u24 = F.upsample(up24_heat, size=x.size()[2:], mode='bilinear')
		
        u31 = F.upsample(up31_heat, size=x.size()[2:], mode='bilinear')
	u32 = F.upsample(up32_heat, size=x.size()[2:], mode='bilinear')
	u33 = F.upsample(up33_heat, size=x.size()[2:], mode='bilinear')
	u34 = F.upsample(up34_heat, size=x.size()[2:], mode='bilinear')
		
        u41 = F.upsample(up41_heat, size=x.size()[2:], mode='bilinear')
	u42 = F.upsample(up42_heat, size=x.size()[2:], mode='bilinear')
	u43 = F.upsample(up43_heat, size=x.size()[2:], mode='bilinear')
	u44 = F.upsample(up44_heat, size=x.size()[2:], mode='bilinear')
		
	infer_result = F.sigmoid(pred0_1) + F.sigmoid(pred0_2) +\
	               F.sigmoid(d11) + F.sigmoid(d12) + F.sigmoid(d13) + F.sigmoid(d14) +\
		       F.sigmoid(u11) + F.sigmoid(u12) + F.sigmoid(u13) + F.sigmoid(u14) +\
		       F.sigmoid(d21) + F.sigmoid(d22) + F.sigmoid(d23) + F.sigmoid(d24) +\
		       F.sigmoid(u21) + F.sigmoid(u22) + F.sigmoid(u23) + F.sigmoid(u24) +\
		       F.sigmoid(d31) + F.sigmoid(d32) + F.sigmoid(d33) + F.sigmoid(d34) +\
		       F.sigmoid(u31) + F.sigmoid(u32) + F.sigmoid(u33) + F.sigmoid(u34) +\
		       F.sigmoid(d41) + F.sigmoid(d42) + F.sigmoid(d43) + F.sigmoid(d44) +\
		       F.sigmoid(u41) + F.sigmoid(u42) + F.sigmoid(u43) + F.sigmoid(u44)
		
		
		
        if self.training:
            return pred0_1, pred0_2, \
                   d11, d12, d13, d14, u11, u12, u13, u14,\
	           d21, d22, d23, d24, u21, u22, u23, u24,\
	           d31, d32, d33, d34, u31, u32, u33, u34,\
	           d41, d42, d43, d44, u41, u42, u43, u44
        else:
            return infer_result
			
class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = in_dim / 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
