from torch import nn
import torch.nn.functional as F

import model.mobilenetv3 as models


class BaselineNet(nn.Module):
    def __init__(self, mode='large', width_mult=1., dropout=0.1, classes=2, zoom_factor=8,
                 atrous3=False, atrous4=False,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(BaselineNet, self).__init__()
        assert mode in ['large', 'small']
        if mode == 'large':
            fea_dim = int(160 * width_mult)
        elif mode == 'small':
            raise NotImplementedError('Not support mobilenetv3 small now')
            fea_dim = int(96 * width_mult)
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion

        if mode == 'large':
            mobilenet = models.build_mobilenetv3_large(pretrained=pretrained, width_mult=width_mult)
        else:
            raise NotImplementedError('Not support small mobilenetv3')

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = (
            mobilenet.layer0, mobilenet.layer1, mobilenet.layer2,
            mobilenet.layer3, mobilenet.layer4
        )

        if mode == 'large':
            if atrous3:
                for n, m in self.layer3.named_modules():
                    if 'conv.3' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)

            if atrous4:
                for n, m in self.layer4.named_modules():
                    if 'conv.3' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (4, 4), (1, 1)


        else:
            raise RuntimeError('Not support small mobilenetv3')

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(128, classes, kernel_size=1)
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
        # assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        x = self.cls(x)
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


if __name__ == '__main__':
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473)
    model = BaselineNet(mode='large', width_mult=1.0, dropout=0.1, classes=19, zoom_factor=8,
                        atrous3=True, atrous4=True, pretrained=True)
    model.eval()
    print(model)
    output = model(input)
    print('Baseline', output.size())
