from torch import nn
import torch.nn.functional as F
import model.mobilenetv3 as models
import torch

__all__ = ['MLMSNetv2']


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channel, make_divisible(channel // reduction, 8), 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(make_divisible(channel // reduction, 8), channel, 1, 1, 0, bias=False),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation, groups=inplanes, bias=False),
            nn.BatchNorm2d(inplanes),
            h_swish(),
            # pw
            nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self._init_weight()

    def forward(self, x):
        return self.conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MSF(nn.Module):
    def __init__(self, cin, strides=[6, 12, 18], up=1):
        super(MSF, self).__init__()
        blocks = []
        for step in strides:
            blocks.append(_ASPPModule(cin, cin, 3, padding=step, dilation=step, BatchNorm=nn.BatchNorm2d))
        self.blocks = nn.ModuleList(blocks)
        self.cin = cin
        self.all_channel = cin * (1 + len(strides))
        self.atten = SELayer(self.all_channel)
        self.fuse = ConvBNReLU(self.all_channel, self.cin * up, ks=1, stride=1, padding=0)
        self._init_weight()

    def forward(self, x):
        feats = [x]
        for block in self.blocks:
            feats.append(block(x))
        feats = torch.cat(feats, dim=1)
        feats = self.atten(feats)
        return self.fuse(feats)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MLF(nn.Module):
    def __init__(self, cin):
        super(MLF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.atten = SELayer(cin)
        self.fuse = ConvBNReLU(cin, cin, ks=1, stride=1, padding=0)
        self._init_weight()

    def forward(self, x):
        feat = x.clone()
        atten = self.atten(x)
        feat += atten
        out = self.fuse(feat)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MLMSNetv2(nn.Module):
    def __init__(self, mode='large', width_mult=1., dropout=0.1, classes=2, zoom_factor=8, use_msf=True, use_mlf=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, large=False):
        super(MLMSNetv2, self).__init__()
        assert mode in ['large', 'small']
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.use_msf = use_msf
        self.use_mlf = use_mlf

        if mode == 'large':
            mobilenet = models.build_mobilenetv3_large(pretrained=pretrained, width_mult=width_mult)
            self.cins = [40, 112, 160]
            for i, cin in enumerate(self.cins):
                self.cins[i] = make_divisible(cin * width_mult, 8)
        elif mode == 'small':
            mobilenet = models.build_mobilenetv3_small(pretrained=pretrained, width_mult=width_mult)
            self.cins = [24, 48, 96]
            for i, cin in enumerate(self.cins):
                self.cins[i] = make_divisible(cin * width_mult, 8)
        else:
            raise NotImplementedError('not support mode {}'.format(mode))

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = (
            mobilenet.layer0, mobilenet.layer1, mobilenet.layer2,
            mobilenet.layer3, mobilenet.layer4
        )
        up = 2 if large else 1
        if use_msf:
            self.msf0 = MSF(self.cins[0], up=up)
            self.msf1 = MSF(self.cins[1], up=up)
            self.msf2 = MSF(self.cins[2], up=up)
        fea_dim = sum(self.cins) * up
        if use_mlf:
            self.mlf = MLF(fea_dim)
        self.cls = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(fea_dim, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feats = []
        if self.use_msf:
            feats.append(self.msf0(x))
        else:
            feats.append(x)

        x_tmp = self.layer3(x)
        if self.use_msf:
            feats.append(self.msf1(x_tmp))
        else:
            feats.append(x_tmp)
        x = self.layer4(x_tmp)
        if self.use_msf:
            feats.append(self.msf2(x))
        else:
            feats.append(x)

        feat = torch.cat(feats, dim=1)
        if self.use_mlf:
            feat = self.mlf(feat)

        x = self.cls(feat)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(x, y)
            return x.max(1)[1], main_loss, None
        else:
            return x

    def get_train_params(self):
        org_params = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        new_params = [self.cls]
        if self.use_msf:
            new_params.extend([self.msf0, self.msf1, self.msf2])
        if self.use_mlf:
            new_params.extend([self.mlf])
        return org_params, new_params


if __name__ == '__main__':
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 713, 713)
    model = MLMSNetv2(mode='large', width_mult=1.0, dropout=0.1, classes=21, zoom_factor=8,
                      use_msf=True, use_mlf=True, pretrained=False)
    model.eval()
    print(model)
    output = model(input)
    print('MLMSv2', output.size())
