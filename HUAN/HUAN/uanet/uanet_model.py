# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from torch import nn
from .uanet_parts import *

class UANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UANet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.down1_heat = nn.Sequential(
           # nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.down2_heat = nn.Sequential(
           # nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.down3_heat = nn.Sequential(
           # nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.down4_heat = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )

        self.up4 = up(512, 512, n_classes)
        self.up3 = up(512, 256, n_classes)
        self.up2 = up(256, 128, n_classes)
        self.up1 = up(128, 64, n_classes)

        self.up4_heat = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        self.up3_heat = nn.Sequential(
           # nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        self.up2_heat = nn.Sequential(
           # nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        self.up1_heat = nn.Sequential(
           # nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x = self.inc(x)

        x = self.down1(x)
        down1_heat = self.down1_heat(x)
        x = self.down2(x)
        down2_heat = self.down2_heat(x)
        x = self.down3(x)
        down3_heat = self.down3_heat(x)
        x = self.down4(x)
        down4_heat = self.down4_heat(x)
       
        x = self.up4(x, 512, down4_heat)
        up4_heat = self.up4_heat(x)
        x = self.up3(x, 256, down3_heat)
        up3_heat = self.up3_heat(x)
        x = self.up2(x, 128, down2_heat)
        up2_heat = self.up2_heat(x)
        x = self.up1(x, 64, down1_heat)
        up1_heat = self.up1_heat(x)

        x = self.outc(x)

        if self.training:
            return F.sigmoid(x), F.sigmoid(up1_heat), F.sigmoid(up2_heat), F.sigmoid(up3_heat), F.sigmoid(up4_heat), \
               F.sigmoid(down1_heat), F.sigmoid(down2_heat), F.sigmoid(down3_heat), F.sigmoid(down4_heat)
        else:
            return (F.sigmoid(x)+F.sigmoid(F.upsample(up1_heat,size=x.size()[2:],mode='bilinear'))+F.sigmoid(F.upsample(up2_heat,size=x.size()[2:],mode='bilinear'))+F.sigmoid(F.upsample(up3_heat,size=x.size()[2:],mode='bilinear'))+F.sigmoid(F.upsample(up4_heat,size=x.size()[2:],mode='bilinear')))/5
       # else:
       #     return F.sigmoid(x) 
