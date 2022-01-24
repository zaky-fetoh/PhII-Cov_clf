import torch.nn.functional as f
import torch.nn as nn
import torch as t


class switching_w(nn.Module):
    def __init__(self, inchannels, ochannels=3):
        super(switching_w, self).__init__()
        indim = 16
        self.conv1 = nn.Conv2d(inchannels, indim,
                               kernel_size=5,
                               padding=2)
        self.bn = nn.BatchNorm2d(indim)
        self.id_path = nn.Conv2d(indim, inchannels, 1)
        self.wai = nn.Conv2d(inchannels, ochannels,
                             kernel_size=5,
                             padding=2, )

    def forward(self, X):
        X = self.id_path(f.leaky_relu(self.bn(self.conv1(X)))) + X
        X = self.wai(f.leaky_relu(X))
        return f.softmax(X, 1).unsqueeze(-1).permute(1, 0, 4, 2, 3)


class waspp(nn.Module):
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 2, 3, ]):
        super(waspp, self).__init__()
        indim = 32
        self.conv1 = nn.Conv2d(inchannels, indim, kernel)
        self.bn1 = nn.BatchNorm2d(indim)
        self.conv2 = nn.Conv2d(indim, indim, kernel)
        self.bn2 = nn.BatchNorm2d(indim)
        self.conv3 = nn.ModuleList([nn.Conv2d(indim, ochannels, 1) for _ in range(rates.__len__())])

        self.proj = nn.Conv2d(inchannels, ochannels, 1)
        self.sw = switching_w(inchannels, ochannels=rates.__len__() + 1)
        self.pading = [(rate * (kernel - 1)) // 2 for rate in rates]
        self.dilations = rates

    def forward(self, X):
        w = self.sw(X)
        self.conv1.dilation = self.dilations[0]
        self.conv1.padding = self.pading[0]
        self.conv2.dilation = self.dilations[0]
        self.conv2.padding = self.pading[0]
        acc = self.conv3[0](f.leaky_relu(self.bn2(
            self.conv2(f.leaky_relu(self.bn1(
                self.conv1(X))))))) * w[0]
        for i in range(1, 3):
            self.conv1.dilation = self.dilations[i]
            self.conv1.padding = self.pading[i]
            self.conv2.dilation = self.dilations[i]
            self.conv2.padding = self.pading[i]
            acc += self.conv3[i](f.leaky_relu(self.bn2(
                self.conv2(f.leaky_relu(self.bn1(
                    self.conv1(X))))))) * w[i]
        acc += self.proj(X) * w[3]
        return acc

class b_switching(nn.Module):
    def __init__(self, inchannels, ochannels=3):
        super(b_switching, self).__init__()
        self.conv = nn.Conv2d(inchannels, inchannels,
                              kernel_size=3,
                              padding=1)
        self.fc = nn.Conv2d(inchannels, ochannels, 1)

    def forward(self, X):
        X = self.conv(X)
        b, d, x, y = X.shape
        X = f.avg_pool2d(X, (x, y))
        X = self.fc(X)
        return f.softmax(X,1).unsqueeze(-1).permute(1,0,*range(2,5))


class hwaspp(nn.Module):
    def __init__(self,inchannels, ochannels,
                 kernel=3,num=5):
        super(hwaspp, self).__init__()
        self.waspps = nn.ModuleList([waspp(inchannels,ochannels,
                                           kernel)for _ in range(num)])
        self.addRis = False
        if inchannels == ochannels :
            self.addRis=True
        self.bsw = b_switching(inchannels, num if not self.addRis else num+1 )
        self.num=num
    def forward(self, X):
        wai = self.bsw(X)
        acc = self.waspps[0](X) * wai[0]
        for i in range(1,self.num):
            acc += self.waspps[i](X) * wai[i]
        if self.addRis :
            acc += X * wai[-1]
        return acc



if __name__ == '__main__':
    inp = t.randn(5, 10, 20, 20)
    saspp = hwaspp(10, 10)
    out = saspp(inp)
