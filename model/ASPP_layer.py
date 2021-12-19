import torch.nn.functional as f
import torch.nn as nn
import torch as t


class sconv2D(nn.Module):
    # depth wise Separable kernel with diff rate
    #only square kernels dto padding
    def __init__(self, inchannels, ochannels, kernals, rate):
        super(sconv2D, self).__init__()
        d = (rate * (kernals-1))//2
        print(d)
        self.conv = nn.Conv2d(inchannels, 1, kernals,
                              padding= d, dilation=rate, )
        self.bn = nn.BatchNorm2d(1)
        self.dw = nn.Conv2d(1, ochannels,1 )

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.dw(X)
        return X


class ASPP(nn.Module):
    # Atrous Spatial Prymaid Pooling w same kernel size
    # out channels shoud be dividable by4
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 4, 8, 12, ]):
        super(ASPP, self).__init__()
        pbo = ochannels // 4
        self.mlis = nn.ModuleList()
        for r in rates:
            self.mlis.append(sconv2D(inchannels, pbo, kernel, r))

    def forward(self, X):
        bff = list()
        for lyr in self.mlis:
            bff.append(lyr(X))
        return t.cat(bff, 1, )

class global_context(nn.Module):
    #presive global Context f the inp
    def __init__(self,inchannels):
        super(global_context, self).__init__()
        self.fc = nn.Conv2d(inchannels,inchannels,1)
        self.bn = nn.BatchNorm2d(inchannels)
    def forward(self, X):
        b, d, x, y = X.shape
        X = f.avg_pool2d(X,(x,y),)
        att = t.sigmoid(self.bn(self.fc(X)))
        return X * att

class ASPPwGlobal_cont(nn.Module):
    ##ASPP with Global Context
    def __init__(self, inchannels, ochannels,
                 kernel=3, rates=[1, 4, 8, 12,],
                 actfunc = f.leaky_relu):
        super(ASPPwGlobal_cont, self).__init__()
        self.aspp  = ASPP(inchannels,ochannels,kernel,rates)
        self.bn = nn.BatchNorm2d(ochannels)
        self.actfunc = actfunc

        self.gc = global_context(inchannels)
        self.fc = nn.Conv2d(inchannels, ochannels, 1)
    def forward(self, X):
        sp = self.actfunc(self.bn(self.aspp(X)))
        glbc = self.fc(self.gc(X))
        return sp * glbc

if __name__ == '__main__':
    inp = t.Tensor(13, 3,15,15)
    assp = ASPPwGlobal_cont(3,4*6)
    out = assp(inp)
    print(out.shape)
