import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16_bn

# initial all the parameters in network
# for Convolotional and FC layer, normal
# bias set to zero
# BatchNorm: ...
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# for convolution operation, set kernel_size=3 and padding=1
# for deconvolution operation, set kernel_size=2 and stride=2
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers, size=None):
        super(_DecoderBlock, self).__init__()

        middle_channels = int( in_channels / 2)
        layers = [
            #nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            #意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            #这样能够节省运算内存，不用多存储其他变量
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)

class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SegNet, self).__init__()
        vgg = vgg16_bn(pretrained=pretrained)

        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:24])
        self.enc4 = nn.Sequential(*features[24:34])
        self.enc5 = nn.Sequential(*features[34:44])

        self.dec5 = _DecoderBlock(512, 512, 3)
        self.dec4 = _DecoderBlock(1024, 256, 3)
        self.dec3 = _DecoderBlock(512, 128, 3)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights( self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        up_sampled5 = F.upsample(enc5, size= tuple((torch.tensor(enc4.shape[2:])).numpy()))
        dec5 = self.dec5(up_sampled5)

        merge4 = torch.cat([enc4,dec5],1)
        up_sampled4 = F.upsample(merge4, size= tuple((torch.tensor(enc3.shape[2:])).numpy()))
        dec4 = self.dec4(up_sampled4)

        merge3 = torch.cat([enc3, dec4], 1)
        up_sampled3 = F.upsample(merge3, size=tuple((torch.tensor(enc2.shape[2:])).numpy()))
        dec3 = self.dec3(up_sampled3)

        merge2 = torch.cat([enc2, dec3], 1)
        up_sampled2 = F.upsample(merge2, size=tuple((torch.tensor(enc1.shape[2:])).numpy()))
        dec2 = self.dec2(up_sampled2)

        merge1 = torch.cat([enc1, dec2], 1)
        up_sampled1 = F.upsample(merge1, size=tuple((torch.tensor(x.shape[2:])).numpy()))
        dec1 = self.dec1(up_sampled1)

        return dec1