import torch.nn.functional as f
import torch.nn as nn
from model import*


class CNN (nn.Module):
    def __init__(self, inshape = 1, clsnum=2,
                 cdepth =[32, 64, 128, 256, 32, 32],
                 fdepth =[128,64,], SPP_lvl = 10,
                 cfx = Conv2DLinSepKer,
                 ):
        super(CNN, self).__init__()
        self.convbase = nn.ModuleList()
        self.densebase = nn.ModuleList()
        self.convbase.append(nn.BatchNorm2d(inshape))#######
        inps,pc = inshape, 3
        for d in cdepth:
            self.convbase.append(DConv2d(inps,d,fx=cfx))
            if pc :
                self.convbase.append(nn.MaxPool2d(2))
                pc-=1
            inps = d
        self.convbase.append(Single_level_SSP2D(SPP_lvl))
        inps = self.convbase[-1].SSP.tbins *d

        self.densebase.append(nn.Dropout(.5))
        for d in fdepth :
            self.densebase.append(nn.Linear(inps, d))
            self.densebase.append(nn.LeakyReLU())
            inps = d
        self.densebase.append(nn.Linear(d, clsnum))
        self.convbase= nn.Sequential(*self.convbase)
        self.densebase = nn.Sequential(*self.densebase)
    def forward(self, X):
        return self.densebase(self.convbase(X))


if __name__ == '__main__':
    pass
