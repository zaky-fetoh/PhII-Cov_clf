import torch.nn.functional as f
import torch.nn as nn
import torch as t


def get_padding(kernel_size):
    kernel_size = kernel_size if isinstance(kernel_size,
                            tuple) else (kernel_size,kernel_size)
    padding = [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
               for k in kernel_size[::-1]]
    pads = [padding[i][j] for i in range(2) for j in range(2)]
    return pads


class SameConv2d(nn.Module):
    def __init__(self, in_chunnels, out_chunnels, kernel_size=3, group=1):
        super().__init__()
        self.in_chunnels, self.out_chunnels, self.kernel_size, self.group = in_chunnels, out_chunnels, kernel_size, group
        self.conv = nn.Conv2d(in_chunnels, out_chunnels, kernel_size, groups=group)
        self.pads = get_padding(self.kernel_size)

    def forward(self, X):
        X = f.pad(X, self.pads)
        return self.conv(X)


class Conv2DLinSepKer(nn.Module):
    def __init__(self, in_chunnels, out_chunnels, kernel_len=3):
        super(Conv2DLinSepKer, self).__init__()
        self.in_chunnels, self.out_chunnels, self.kernel_len = in_chunnels, out_chunnels, kernel_len

        self.actfunc = f.leaky_relu

        self.X_conv = SameConv2d(in_chunnels, in_chunnels, (kernel_len, 1), in_chunnels)
        self.Y_conv = SameConv2d(in_chunnels, in_chunnels, (1, kernel_len), in_chunnels)
        self.pointwise = nn.Conv2d(in_chunnels, out_chunnels, 1)

    def forward(self, X):
        for ly in [self.X_conv, self.Y_conv, self.pointwise]:
            X = self.actfunc(ly(X))
        return X

#
#
# class DConv2d(nn.Module):
#     #modified : used addition instead f cating
#     def __init__(self,in_chunnels, out_chunnel,
#                  kernels = [3, 5, 7, ]*3,
#                  fx = Conv2DLinSepKer):
#         super(DConv2d, self).__init__()
#         self.lyrs = nn.ModuleList()
#         self.actfunc = f.leaky_relu
#         ips = in_chunnels
#         for lk in kernels:
#             self.lyrs += [nn.Sequential(
#                 fx(ips, out_chunnel, lk),
#                 nn.BatchNorm2d(out_chunnel),
#             )]
#             ips = out_chunnel
#     def forward(self, X):
#         #print(X.shape)
#         X = self.actfunc(self.lyrs[0](X))
#         for lyr in self.lyrs[1:]:
#             o = self.actfunc(lyr(X))
#             X += o
#         return o


class DConv2d(nn.Module):
    def __init__(self,in_chunnels, out_chunnel,
                 kernels = [3,  ]*4,
                 fx = Conv2DLinSepKer):
        super(DConv2d, self).__init__()
        self.lyrs = nn.ModuleList()
        self.actfunc = f.leaky_relu
        ips = in_chunnels
        for lk in kernels:
            self.lyrs += [nn.Sequential(
                fx(ips, out_chunnel, lk),
                nn.BatchNorm2d(out_chunnel),
            )]
            ips += out_chunnel
    def forward(self, X):
        #print(X.shape)
        for lyr in self.lyrs:
            o = self.actfunc(lyr(X))
            X = t.cat([X,o], 1)
        return o

if __name__ == '__main__':
    dconv = DConv2d(3,10,fx=SameConv2d)
    tens= t.randn(1,3,25,25)
    out  =dconv(tens)







