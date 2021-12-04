import torchvision.transforms as trans
import torchvision
import numpy as np
import cv2.cv2 as cv


def norma(x):
    x -= x.min()
    x /= x.max()
    return x



class Freq_Jitter(object):
    def __init__(self, mu=1, sig=.3,
                 multip=True):
        self.mu, self.sig = mu, sig
        self.eq = np.multiply if multip else np.add

    def __call__(self, im):
        im = np.fft.fft2(im)
        no = np.random.randn(*im.shape)
        no = self.mu + (no * self.sig)
        return np.fft.ifft2(self.eq(im, no)).real


class RandScale(object):
    def __init__(self, scales=[224 - 50, 224 - 25,
                               224, 224 + 25, 224 + 50]):
        self.scales = scales
        self.new_scale()
        self._countinit()

    def __call__(self, im):
        self._count()
        d = self.scales[self.n]
        return cv.resize(im, (d, d))

    def new_scale(self):
        self.n = np.random.randint(0, len(self.scales))

    def frez_count(self):
        self.count_state = False
    def unfrez_count(self):
        self.count_state = True


    def _countinit(self):
        self.ScaleHist = np.ones((self.scales.__len__(),))
        self.total = self.scales.__len__()
        self.count_state = True


    def _count(self):
        if self.count_state:
            self.ScaleHist[self.n] += 1
            self.total += 1


class reform(object):
    def __call__(self, im):
        return np.asarray(norma(im[..., np.newaxis,]) * 255,
                          dtype=np.uint8)


def get_aug():
    return trans.Compose([
        Freq_Jitter(), RandScale(),
        reform(), trans.ToPILImage(),
        trans.RandomRotation(20),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
        norma,
    ]), 1


if __name__ == '__main__':
    pth = 'E:/covdt_CoCo/DATASETS/QaTa-COV19/QaTa-COV19/Images/covid_1.png'
    import matplotlib.pyplot as plt

    im = plt.imread(pth)
    aug, s = get_aug()
    for i in range(10):
        new_im = aug(im)
        print(new_im.shape)
        aug.transforms[s].new_scale()
