import torch.nn.functional as f
import torch.nn as nn
import torch as t


class switching_w(nn.Module):
    def __init__(self, inchannels, ochannels=3):
        super(switching_w, self).__init__()
        self.conv = nn.Conv2d(inchannels, inchannels,
                              kernel_size=5,
                              padding=2)
        self.bn = nn.BatchNorm2d(inchannels)
        self.wai = nn.Conv2d(inchannels, ochannels,
                            kernel_size=5,
                            padding=2,)

    def forward(self, X):
        X = f.leaky_relu(self.bn(self.conv(X)))
        X = self.wai(X)
        return f.softmax(X,1).unsqueeze(-1).permute(1,0,4,2,3)


class waspp(nn.Module):
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 2, 3, ]):
        super(waspp, self).__init__()
        self.conv = nn.Conv2d(inchannels, ochannels, kernel)
        self.sw = switching_w(inchannels, ochannels= rates.__len__())
        self.pading = [(rate * (kernel - 1)) // 2 for rate in rates]
        self.dilations = rates

    def forward(self, X):
        w = self.sw(X)
        self.conv.dilation = self.dilations[0]
        self.conv.padding = self.pading[0]
        acc = self.conv(X) * w[0]
        for i in range(1,3):
            self.conv.dilation = self.dilations[i]
            self.conv.padding = self.pading[i]
            acc += self.conv(X) * w[i]
        return acc
if __name__ == '__main__':
    inp = t.randn(5, 10, 20, 20)
    saspp = waspp(10, 15)
    out = saspp(inp)







