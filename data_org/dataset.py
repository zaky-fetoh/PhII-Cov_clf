from matplotlib.pyplot import imread
import torch.utils.data as data
from data_org import *
from glob import glob
import torch as t
import os


def loadfile_name(pth, ext='.png'):
    outlis = list()
    for item in glob(pth + '*' + ext):
        outlis.append(
            os.path.join(pth, item)
        )
    return outlis


ENCODE = {'Normal': 0, 'Covid': 1}


class covdata(data.Dataset):
    # Raw Dataset
    def __init__(self,
                 Covidpath=COVID_DIR_PATH,
                 Normalpath=NORMAL_DIR_PATH,
                 balance=True):
        # super(covdata, self).__init__()
        self.labels = {0: 'Normal', 1: 'Covid'}
        self.items = list()
        covs = loadfile_name(Covidpath)
        nors = loadfile_name(Normalpath)
        m = min([len(x) for x in [covs, nors]])
        for i, ds in enumerate([nors, covs, ]):
            ds = ds[:m] if balance else ds
            self.items += [{'im_path': item,
                            'label': self.labels[i]} for item in ds]
        # print(self.__len__())

    def __len__(self):
        return self.items.__len__()

    def __getitem__(self, i):
        return self.items[i]


class QaTaCov(data.Dataset):
    def __init__(self, aug):
        self.pathset = covdata()
        self.aug = aug

    def __len__(self):
        return self.pathset.__len__()

    def __getitem__(self, item):
        obj = self.pathset.__getitem__(item)
        im = imread(obj['im_path'])
        lb = ENCODE[obj['label']]
        return self.aug(im), lb


def create_loader(dts, bs, ):
    tdlr = data.DataLoader(dts, bs, shuffle=True,
                           pin_memory=True, num_workers=2,
                           )
    return tdlr


def getloaders(aug, trtes=[.7, .1, .2], bs=128):
    dts = QaTaCov(aug)
    tl = dts.__len__()
    slen = [int(x * tl) for x in trtes[:2]]
    slen.append(tl - sum(slen))
    tr, va, te = data.random_split(dts, slen)
    return [create_loader(x, bs) for x in [tr, va, te]]


if __name__ == '__main__':
    lis = loadfile_name(COVID_DIR_PATH)
