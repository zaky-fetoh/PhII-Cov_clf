import cv2.cv2 as cv
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt


def get_areas(ims):
    ret, thresh = cv.threshold(ims, 0, 255, cv.THRESH_BINARY)
    out = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
    return out[2][1:, 4].tolist()


def gethistarr(dic):
    arr = np.zeros((max(dic.keys()) + 1,))
    for k, v in dic.items():
        arr[k] = v
    return arr


def avg(arr, w=3):
    out = np.zeros_like(arr)
    for i in range(out.shape[0]):
        out[i] = arr[i - w:i + w + 1].sum()
    return out


class ScaleHist(object):
    def __init__(self, masks_path,
                 oscale=224,
                 scales=[150 - 50, 224 - 25,
                         224, 224 + 50, 512 ]):
        self.path = masks_path
        self.single_scale_hist = dict()
        self.multiscale_hist = dict()
        self.scales = scales
        self.oscale = oscale

    def update_single_scale_hist(self, area):
        if area in self.single_scale_hist:
            self.single_scale_hist[area] += 1
        else:
            self.single_scale_hist[area] = 1

    def update_multiscale_hist(self, area):
        if area in self.multiscale_hist:
            self.multiscale_hist[area] += 1
        else:
            self.multiscale_hist[area] = 1

    def getpaths(self):
        pth = self.path + r'\*.png'
        outlis = list()
        for item in glob(pth):
            outlis.append(
                os.path.join(pth, item)
            )
        return outlis

    def fit(self):
        i = 0
        ims_path = self.getpaths()
        for ipth in ims_path:
            print(i, end=' ')
            i += 1
            im = cv.imread(ipth, 0)

            ssim = cv.resize(im, (self.oscale, self.oscale))
            for a in get_areas(ssim):
                self.update_single_scale_hist(a)

            for sc in self.scales:
                msim = cv.resize(im, (sc, sc))
                for a in get_areas(msim):
                    self.update_multiscale_hist(a)

    def get_nphist(self):
        return [gethistarr(d) for d in [self.single_scale_hist,
                                        self.multiscale_hist,
                                        ]]


def scat(ar,lbl):
    plt.figure()
    plt.scatter(np.arange(ar.shape[0])[np.where(ar != 0)],
                ar[np.where(ar != 0)],s = 100,
                alpha=.1,)
    plt.xlabel('scale')
    plt.ylabel('freq')
    plt.title(lbl)

def gethistfreq(arr, scales_perbin = 1000):
    lis= list()
    i = 0
    while i < arr.shape[0]:
        lis.append(arr[i:i+scales_perbin].sum())
        i += scales_perbin
    return np.array(lis)



if __name__ == '__main__':
    imp = r'E:\covdt_CoCo\DATASETS\QaTa-COV19\QaTa-COV19\Ground-truths'
    sh = ScaleHist(imp)
    sh.fit()
    shis, mulhis = [avg(x,0)for x in sh.get_nphist()]


    shistf, mulhistf = [gethistfreq(x,1000) for x in (shis, mulhis)]
    plt.plot(shistf, label = 'single_scale')
    # plt.legend()
    # plt.figure( )
    plt.plot(mulhistf,label = 'multiscale')
    plt.legend()
    # scat(shis,'single scale')
    # scat(mulhis, 'multiscale')

