import torch
import torch.nn as nn

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k,p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c1, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    
    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
        , dim=1))

class CSPDarknet(nn.Module):
    def __init__(self, base_channel, base_depth):
        super().__init__()

        self.stem = Conv(3, base_channel,6, 2, 2)

        self.dar2 = nn.Sequential(
            Conv(base_channel, base_channel * 2, 3, 2),
            C3(base_channel * 2, base_channel * 2, base_depth)
        )

        self.dar3 = nn.Sequential(
            Conv(base_channel * 2, base_channel * 4, 3, 2),
            C3(base_channel * 4, base_channel * 4, base_depth * 2)
        )

        self.dar4 = nn.Sequential(
            Conv(base_channel * 4, base_channel * 8, 3, 2),
            C3(base_channel * 8, base_channel * 8, base_depth * 3)
        )

        self.dar5 = nn.Sequential(
            Conv(base_channel * 8, base_channel * 16, 3, 2),
            C3(base_channel * 16, base_channel * 16, base_depth * 1),
            SPPF(base_channel * 16, base_channel * 16)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dar2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dar3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dar4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dar5(x)
        feat3 = x
        return feat1, feat2, feat3
    

if __name__ == "__main__":
    csp_model = CSPDarknet(64, 4)