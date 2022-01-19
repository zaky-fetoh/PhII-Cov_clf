import matplotlib.pyplot as plt
import numpy as np
import pickle

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

def load_train_hist(file_name):
    with open(file_name, 'rb')as file:
        hist = pickle.load(file)
    return hist

acc = load_train_hist(r'E:\PhII Cov_clf\weights\hist47.p')
fltandplt(acc)