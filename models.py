import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from resnext.resnext101_regular import ResNeXt101


class _Feedback(nn.Module):
    def __init__(self):
        super(_Feedback, self).__init__()

        self.fb_block = _Feedback_Block()

        self.conv_com = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 64, 1, padding=0, bias=False), nn.BatchNorm2d(64)),
            nn.Sequential(nn.Conv2d(192, 64, 1, padding=0, bias=False), nn.BatchNorm2d(64)),
        ])

    def forward(self, F_in1, F_in2, fb_imgs, fb_num_steps):
        
        F_in_init = torch.cat((F_in1, F_in2), 1)
        F_in = F_in_init

        for idx in range(fb_num_steps):
            if idx == 0:
                F_in = self.conv_com[0](F_in)
            
            else :
                F_in = self.conv_com[1](F_in)

            out, img = self.fb_block(F_in)
            F_in = torch.cat((F_in_init, out), 1)
            fb_imgs.append(img)
          
        return out


class _Feedback_Block(nn.Module): 
    def __init__(self):
        super(_Feedback_Block, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, dilation=1, bias=False), nn.BatchNorm2d(64),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(64),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=3, dilation=3, bias=False), nn.BatchNorm2d(64),
        )

        self.attention_map = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0, dilation=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, padding=0, dilation=1, bias=False), nn.Sigmoid() 
        )

    def forward(self, x): 
        
        layer0 = self.conv0(F.relu(x, False))
        layer1 = self.conv1(F.relu((x + layer0), False))
        layer2 = self.conv2(F.relu((x + layer0 + layer1), False))
        layer3 = self.attention_map(F.relu((x + layer0 + layer1 + layer2), False))

        out = F.relu((x + layer0 + layer1 + layer2), False)

        return out, layer3

     
class _SE_Block(nn.Module): 
    def __init__(self, channel, reduction=16):
        super(_SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _Spatial_Attention(nn.Module):
    def __init__(self, channel):
        super(_Spatial_Attention, self).__init__()
        self.at1_1 = nn.Sequential(
            nn.Conv2d(64, channel//2, kernel_size=(1,9), stride=(1,1), padding=(0,4)),
            nn.BatchNorm2d(channel//2), 
            nn.ReLU()
        )
        self.at1_2 = nn.Sequential(
            nn.Conv2d(channel//2, 1, kernel_size=(9,1), stride=(1,1), padding=(4,0)),
            nn.ReLU()
        )

        self.at2_1 = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=(9,1), stride=(1,1), padding =(4,0)),
            nn.BatchNorm2d(channel//2), 
            nn.ReLU()
        )
        self.at2_2 = nn.Sequential(
            nn.Conv2d(channel//2, 1, kernel_size=(1,9), stride=(1,1), padding=(0,4)),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.at1_1(x)
        x1 = self.at1_2(x1)
        x2 = self.at2_1(x)
        x2 = self.at2_2(x2)
        x_add = torch.add(x1, x2)
        weight = self.sigmoid(x_add)
        
        return weight*x

class _Up(nn.Module):
    def __init__(self, in_channel, out_channel, mode='relu'):
        super(_Up, self).__init__()

        if mode == 'relu':
            self.dconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(out_channel), nn.ReLU()
            )
        
        elif mode == 'sig':
            self.dconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(out_channel), nn.Sigmoid()
            )

    def forward(self, x):
        layer0 = self.dconv(x)

        return layer0

class _Fusion_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(_Fusion_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0), nn.Sigmoid()
        )
        
    def forward(self, x):
        layer0 = self.conv(x)

        return layer0


class AFFPNet(nn.Module): 
    def __init__(self, fb_num_steps):
        super(AFFPNet, self).__init__()
        resnext = ResNeXt101()

        self.fb_num_steps = fb_num_steps

        self.resnext_layer0 = resnext.layer0 
        self.resnext_layer1 = resnext.layer1
        self.resnext_layer2 = resnext.layer2
        self.resnext_layer3 = resnext.layer3
        self.resnext_layer4 = resnext.layer4

        #channel compression 
        self.cdown1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())  
        self.cdown2 = nn.Sequential(nn.Conv2d(512, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())  
        self.cdown3 = nn.Sequential(nn.Conv2d(1024, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()) 
        self.cdown4 = nn.Sequential(nn.Conv2d(2048, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()) 

        #channel attention
        self.ca_x4 = _SE_Block(channel=64, reduction=16) 
        self.ca_x3 = _SE_Block(channel=64, reduction=16)

        #spatial_attention
        self.sa_x2 = _Spatial_Attention(channel=64)
        self.sa_x1 = _Spatial_Attention(channel=64)
        
        self.fb_x1 = _Feedback()
        self.fb_x2 = _Feedback()
        self.fb_x3 = _Feedback()

        self.f_conv = _Fusion_conv(fb_num_steps*3+1, 1)

        self.upconv_x4 = _Up(in_channel=64, out_channel=64, mode='relu')
        self.upconv_x3 = _Up(in_channel=64, out_channel=64, mode='relu')
        self.upconv_x2 = _Up(in_channel=64, out_channel=64, mode='relu')
        self.upconv_x1 = _Up(in_channel=64, out_channel=32, mode='relu')
        self.upconv_x0 = _Up(in_channel=32, out_channel=1, mode='sig')
       
    def forward(self, x):
        
        layer0 = self.resnext_layer0(x)        
        layer1 = self.resnext_layer1(layer0) 
        layer2 = self.resnext_layer2(layer1) 
        layer3 = self.resnext_layer3(layer2) 
        layer4 = self.resnext_layer4(layer3) 
        
        layer4_cdown = self.cdown4(layer4)
        layer3_cdown = self.cdown3(layer3)
        layer2_cdown = self.cdown2(layer2)
        layer1_cdown = self.cdown1(layer1)
        
        wfeature_x4 = self.ca_x4(layer4_cdown)
        wfeature_x3 = self.ca_x3(layer3_cdown)
        wfeature_x2 = self.sa_x2(layer2_cdown)
        wfeature_x1 = self.sa_x1(layer1_cdown)

        wfeature_x4_up = self.upconv_x4(wfeature_x4) 

        fb_imgs_x3 = []
        fb_feature_x3 = self.fb_x3(wfeature_x4_up, wfeature_x3, fb_imgs_x3, self.fb_num_steps)


        fb_feature_x3_up = self.upconv_x3(fb_feature_x3) 

        fb_imgs_x2 = []
        fb_feature_x2 = self.fb_x2(fb_feature_x3_up, wfeature_x2, fb_imgs_x2, self.fb_num_steps)

        fb_feature_x2_up = self.upconv_x2(fb_feature_x2) 

        fb_imgs_x1 = []
        fb_feature_x1 = self.fb_x1(fb_feature_x2_up, wfeature_x1, fb_imgs_x1, self.fb_num_steps)

        fb_feature_x1_up = self.upconv_x1(fb_feature_x1) 
        fb_feature_x = self.upconv_x0(fb_feature_x1_up) 

        
        for idx in range (0, self.fb_num_steps):
            fb_imgs_x3[idx] = F.upsample(fb_imgs_x3[idx], size=x.size()[2:], mode='bilinear')
            fb_imgs_x2[idx] = F.upsample(fb_imgs_x2[idx], size=x.size()[2:], mode='bilinear') 
            fb_imgs_x1[idx] = F.upsample(fb_imgs_x1[idx], size=x.size()[2:], mode='bilinear') 

        fb_imgs_cat = torch.cat((fb_imgs_x3[0], fb_imgs_x2[0], fb_imgs_x1[0]), 1)
        for idx in range (1, self.fb_num_steps):
            fb_imgs_cat = torch.cat((fb_imgs_cat, fb_imgs_x3[idx], fb_imgs_x2[idx], fb_imgs_x1[idx]), 1)

        fb_imgs = torch.cat((fb_imgs_cat, fb_feature_x), 1)

        out = self.f_conv(fb_imgs)

        return out, fb_imgs_x1, fb_imgs_x2, fb_imgs_x3
      
        