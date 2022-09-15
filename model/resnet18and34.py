import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch ,stride=1 ,shortcut=None,dilation = 1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
        
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_ch, out_ch, 3, stride=1 , padding=dilation, dilation=dilation,bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):  # 4*1024*1024
    def __init__(self, input_channel):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),  # 64*1024*1024
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64,64*512*512

        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  
        self.layer3 = self.make_layer(128, 256, 6, stride=2,dilation=2)  
        self.layer4 = self.make_layer(256, 512, 3, stride=2,dilation=4) 
        #self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, in_ch, out_ch, block_num, stride=1,dilation=1):

        shortcut = nn.Sequential(  
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut=shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch ,dilation=dilation)) 
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        low_feature = self.layer2(x)  # 28x28x128
        x = self.layer3(low_feature)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        #x = F.avg_pool2d(x, 7)  # 1x1x512
        #x = x.view(x.size(0), -1)  
        #x = self.fc(x)  # 1x1
        return low_feature,x  

class ResNet18(nn.Module):
    def __init__(self, input_channel):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),  # 64*1024*1024
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64,64*512*512

        self.layer1 = self.make_layer(64, 64, 2)  
        self.layer2 = self.make_layer(64, 128, 2, stride=2)  
        self.layer3 = self.make_layer(128, 256, 2, stride=2)  
        self.layer4 = self.make_layer(256, 512, 2, stride=2)  

    def make_layer(self, in_ch, out_ch, block_num, stride=1, padding =0,dilation=1):
        
        shortcut = nn.Sequential(  
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch,padding,dilation))  
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        low_feature = self.layer2(x)  # 28x28x128
        x = self.layer3(low_feature)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        # x = F.avg_pool2d(x, 7)  # 1x1x512
        # x = x.view(x.size(0), -1)  
        # x = self.fc(x)  # 1x1
        return low_feature, x  
