import matplotlib.pyplot as plt
import numpy as np
import pickle


def fltting(trHist):
    tllis, talis = list(), list()
    tsllis, tsalis = list(), list()
    for e, val in trHist.items():
        trl, tra, tsl, tsa = val
        tllis += [np.mean(trl) / 16]
        talis += [np.mean(tra)]
        tsllis += [np.mean(tsl) / 16]
        tsalis += [np.mean(tsa)]
    return tllis, talis, tsllis, tsalis


def load_train_hist(file_name):
    with open(file_name, 'rb')as file:
        hist = pickle.load(file)
    return hist


files = [('finalRes/multiLVLAvpSameKernel.p', 'multilevel SPP avgpooling'),
         ('finalRes/multiLVLMxpSameKernel.p', 'multilevel SPP maxpooling'),
         ('finalRes/singleLVLAvpSameKernel.p', 'single level SPP avgpooling'),
         ('finalRes/singleLVLMxpSameKernel.p', 'single level SPP maxpooling'),
         ('finalRes/ASPPNoGlobalContxt.p', 'ASPP No Global Context'),
         ]
files = [('multiLVLAvpSameKernel.p', 'multilevel SPP avgpooling'),
         ('multiLVLMxpSameKernel.p', 'multilevel SPP maxpooling'),
         ('singleLVLAvpSameKernel.p', 'single level SPP avgpooling'),
         ('singleLVLMxpSameKernel.p', 'single level SPP maxpooling'),
         ('ASPPNoGlobalContxt.p', 'ASPP No Global Context'),
         ('hist401.p', 'swatchable waightedASPP'),
         ('hist97.p', 'spatial waighted ASPP'),
         ]


def plting(train=True):
    plt.subplots(1, 2)
    for p, lb in files:
        if train:
            loss, acc, _, _ = fltting(load_train_hist(p))
        else:
            _, _, loss, acc, = fltting(load_train_hist(p))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label=lb)
        plt.subplot(1, 2, 2)
        plt.plot(loss, label=lb)
    plt.legend()
    plt.subplot(1, 2, 1)
    plt.legend()


if __name__ == '__main__':
    plting(0)
    plting(1)
