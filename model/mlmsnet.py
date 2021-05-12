from torch import nn
import torch.nn.functional as F
import model.mobilenetv3 as models
import torch


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
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

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
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.atten = nn.Conv2d(self.all_channel, self.all_channel, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(self.all_channel)
        self.sigmoid_atten = nn.Sigmoid()
        self.fuse = ConvBNReLU(self.all_channel, self.cin * up, ks=1, stride=1, padding=0)
        self._init_weight()

    def forward(self, x):
        feats = [x]
        for block in self.blocks:
            feats.append(block(x))
        feats = torch.cat(feats, dim=1)
        atten = self.avg_pool(feats)
        atten = self.atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        feats = torch.mul(feats, atten)
        out = self.fuse(feats)
        return out

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
        self.atten = nn.Conv2d(cin, cin, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(cin)
        self.sigmoid_atten = nn.Sigmoid()
        self.fuse = ConvBNReLU(cin, cin, ks=1, stride=1, padding=0)
        self._init_weight()

    def forward(self, x):
        feat = x.clone()
        atten = self.avg_pool(x)
        atten = self.atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        feat += torch.mul(x, atten)
        out = self.fuse(feat)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MLMSNet(nn.Module):
    def __init__(self, mode='large', width_mult=1., dropout=0.1, classes=2, zoom_factor=8, use_msf=True, use_mlf=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, large=False):
        super(MLMSNet, self).__init__()
        assert mode in ['large', 'small']
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.use_msf = use_msf
        self.use_mlf = use_mlf

        if mode == 'large':
            mobilenet = models.build_mobilenetv3_large(pretrained=pretrained, width_mult=width_mult)
            if abs(width_mult - 1.0) < 1e-4:
                self.cins = [40, 112, 160]
        else:
            raise RuntimeError('Not support small mobilenetv3')

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
            nn.Conv2d(fea_dim, 128 * up, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128 * up),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128 * up, classes, kernel_size=1)
        )
        if self.training:
            if mode == 'large':
                fea_in = int(112 * width_mult)

            self.aux = nn.Sequential(
                nn.Conv2d(fea_in, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(64, classes, kernel_size=1)
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
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x

    def get_train_params(self):
        org_params = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        new_params = [self.cls, self.aux]
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
    model = MLMSNet(mode='large', width_mult=1.0, dropout=0.1, classes=21, zoom_factor=8,
                    use_msf=False, use_mlf=False, pretrained=True)
    model.eval()
    print(model)
    output = model(input)
    print('MLMS', output.size())
