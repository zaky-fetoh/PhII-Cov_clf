import torch.nn.functional as F
import torch.nn as nn
import torch


def ceil(x):
    if x == int(x):
        return int(x)
    return int(x) + 1


def get_fv_len(num_ch, splvl):
    return num_ch * Sp_pooling2D.get_total_num_bins(splvl)


class Sp_pooling2D(nn.Module):
    def __init__(self, levels=4):
        super(Sp_pooling2D, self).__init__()
        self.tbins = self.get_total_num_bins(levels)
        self.levels = levels

    @staticmethod
    def get_total_num_bins(levels):
        return sum([x ** 2 for x in range(1, levels + 1)])

    def forward(self, X):
        b, d, x, y = X.shape
        out = F.max_pool2d(X, (x, y)
                           ).view(b, d, -1)
        for i in range(2, self.levels + 1):
            nx, ny = [ceil(k / i) for k in (x, y)]
            sx, sy = x // i, y // i
            out = torch.cat([out,
                             F.avg_pool2d(X, (nx, ny),
                                          stride=(sx, sy)
                                          ).view(b, d, -1)[:,:,:i*i]],
                            -1)
            #print(out.shape)
        return out


class spp2D_layer(nn.Module):
    def __init__(self, levels=5):
        super(spp2D_layer, self).__init__()
        self.Spp = Sp_pooling2D(levels)
        self.levels = levels

    def forward(self, X):
        return self.Spp(X).contiguous().view(X.shape[0], -1)

class Single_level_SSP(nn.Module):
    def __init__(self, level = 20):
        super(Single_level_SSP, self).__init__()
        self.level, self.tbin = level, level**2
    def forward(self, X):
        b, d, x, y = X.shape
        i = self.level
        nx, ny = [ceil(k / i) for k in (x, y)]
        sx, sy = x // i, y // i
        return F.max_pool2d(X, (nx, ny), stride=(sx, sy)
                     ).view(b, d, -1)[:,:,:self.tbin]

class Single_level_SSP2D(nn.Module):
    def __init__(self, level =10):
        super(Single_level_SSP2D, self).__init__()
        self.SSP = Single_level_SSP(level)
    def forward(self, X):
        return self.SSP(X).contiguous().view(X.shape[0], -1)


if __name__ == '__main__':
    for i in range(10,50):
        tens = torch.Tensor(1, 15, i, i)
        sp = spp2D_layer(10)
        out = sp(tens)
