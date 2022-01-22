import torch.nn.functional as f
import torch.nn as nn
import torch as t


class switching_w(nn.Module):
    def __init__(self, inchannels, ochannels=3):
        super(switching_w, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 4,
                              kernel_size=5,
                              padding=2)
        self.bn = nn.BatchNorm2d(4)
        self.id_path = nn.Conv2d(4,inchannels,1)
        self.wai = nn.Conv2d(inchannels, ochannels,
                            kernel_size=5,
                            padding=2,)

    def forward(self, X):
        X = self.id_path(f.leaky_relu(self.bn(self.conv1(X)))) + X
        X = self.wai(f.leaky_relu(X))
        return f.softmax(X,1).unsqueeze(-1).permute(1,0,4,2,3)


class waspp(nn.Module):
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 2, 3, ]):
        super(waspp, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 8, kernel)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, ochannels, 1)

        self.proj = nn.Conv2d(inchannels, ochannels,1)
        self.sw = switching_w(inchannels, ochannels= rates.__len__()+1)
        self.pading = [(rate * (kernel - 1)) // 2 for rate in rates]
        self.dilations = rates

    def forward(self, X):
        w = self.sw(X)
        self.conv1.dilation = self.dilations[0]
        self.conv1.padding = self.pading[0]
        self.conv2.dilation = self.dilations[0]
        self.conv2.padding = self.pading[0]
        acc = self.conv3(f.leaky_relu(self.bn2(
                self.conv2(f.leaky_relu(self.bn1(
                self.conv1(X))))))) * w[0]
        for i in range(1,3):
            self.conv1.dilation = self.dilations[i]
            self.conv1.padding = self.pading[i]
            self.conv2.dilation = self.dilations[i]
            self.conv2.padding = self.pading[i]
            acc += self.conv3(f.leaky_relu(self.bn2(
                self.conv2(f.leaky_relu(self.bn1(
                self.conv1(X)))))))* w[i]
        acc += self.proj(X) * w[3]
        return acc


if __name__ == '__main__':
    inp = t.randn(5, 10, 20, 20)
    saspp = waspp(10, 15)
    out = saspp(inp)







