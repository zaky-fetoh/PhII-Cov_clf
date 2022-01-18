import torch.nn.functional as f
import torch.nn as nn
import torch as t


class switching(nn.Module):
    def __init__(self, inchannels, ochannels=3):
        super(switching, self).__init__()
        self.conv = nn.Conv2d(inchannels, inchannels,
                              kernel_size=3,
                              padding=1)
        self.fc = nn.Conv2d(inchannels, ochannels, 1)

    def forward(self, X):
        X = self.conv(X)
        b, d, x, y = X.shape
        X = f.avg_pool2d(X, (x, y))
        X = self.fc(X)
        return f.softmax(X).unsqueeze(-1).permute(1,0,*range(2,5))


class saspp(nn.Module):
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 2, 4, ]):
        super(saspp, self).__init__()
        self.conv = nn.Conv2d(inchannels, ochannels, kernel)
        self.sw = switching(inchannels)
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
    saspp = saspp(10, 15)
    out = saspp(inp)







