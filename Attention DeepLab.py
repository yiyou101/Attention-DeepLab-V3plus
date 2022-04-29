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
            nn.Conv2d(input_channel, 64, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),  # 64*1024*1024
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64,64*2048*2048
        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。。。
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 第一个stride=2,剩下3个stride=1;28x28x128,128*256*256
        self.layer3 = self.make_layer(128, 256, 6, stride=2,dilation=2)  # 14x14x256;256*128*128
        self.layer4 = self.make_layer(256, 512, 3, stride=2,dilation=4)  # 7x7x512;512*64*64
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
    
    def make_layer(self, in_ch, out_ch, block_num, stride=1,dilation=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut=shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch ,dilation=dilation))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        low_feature = self.shortcut_conv(x)
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        #x = F.avg_pool2d(x, 7)  # 1x1x512
        #x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512
        #x = self.fc(x)  # 1x1
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        return low_feature,x  # 1x1，将结果化为(0~1)之间

#通道注意力
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #做评均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #做最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取，卷积核大小n=k+（k-1）*(d-1)
#   padding和dilation大小一样时，尺寸不变
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_ch)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.branch6 = CBAM(in_ch)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * 5+in_ch, out_ch, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        #第六个分支加上CABM
        att_feature = self.branch6(x)
        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature,att_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class BNDDeepLab(nn.Module):
    def __init__(self, in_ch, num_classes,upsampling=4,backbone="resnet34", downsample_factor=16):
        super(BNDDeepLab, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, num_classes, kernel_size=1, padding=1 // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        if backbone == "resnet34":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = ResNet34(in_ch)
            in_channels = 512
            low_level_channels = 128
        if backbone == "resnet18":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = ResNet18(in_ch)
            in_channels = 512
            low_level_channels = 128
        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(in_ch=in_channels, out_ch=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.low_att = nn.Sequential(
            CBAM(low_level_channels),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(
            SeparableConv2d(
                32 + 256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(32 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        #x = self.conv2d(x)
        #x = self.upsampling(x)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        #low_level_features = self.low_att(low_level_features)
        #low_level_features = self.shortcut_conv(low_level_features)
        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.block(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class SeparableConv2d(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)
