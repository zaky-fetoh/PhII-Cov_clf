import matplotlib.pyplot as plt
import numpy as np

def fltting(trHist):
    tllis, talis= list(),list()
    tsllis, tsalis =list(),list()
    for e, val in trHist.items():
        trl, tra, tsl, tsa = val
        tllis += [np.mean(trl)/16]
        talis += [np.mean(tra)]
        tsllis += [np.mean(tsl)/16]
        tsalis += [np.mean(tsa)]
    return tllis, talis,  tsllis,tsalis


def plting(tr, va,lb='loss'):
    plt.figure()
    plt.plot(tr,label='tr_'+lb)
    plt.plot(va, label='va_'+lb)
    plt.legend()

def fltandplt(trPro):
    trl, tra, val, vaa = fltting(trPro)
    plting(trl,val,'loss')
    plting(tra,vaa, 'acc')

