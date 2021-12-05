import matplotlib.pyplot as plt
import torch.optim as optim
import data_org as dorg
from training import *
import torch.nn as nn
import model as mdl
import torch as t


class Network(object):
    def __init__(self, network=None, augm=None,
                 lders=None, cfx=mdl.Conv2DLinSepKer,
                 bs=32,ustep=128, loss_fn=None, opt=None, cls_num=2,
                 device=t.device('cuda' if t.cuda.is_available() else 'cpu')
                 ):
        self.cfx, self.device, self.cls_num = str(cfx), device, cls_num
        self.network = mdl.CNN(cfx=cfx).to(device=device) if network is None else network
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        self.opt = optim.Adam(self.network.parameters()) if opt is None else opt
        self.augm = dorg.get_aug() if augm is None else augm
        self.trldr, self.valdr, self.teldr = dorg.getloaders(self.augm[0],
                                        bs=bs) if lders is None else lders
        self.ustep = ustep
        self.train_profile = dict()

    def fit(self, epoch=30,startwith=0 ):
        tr_pro = train_and_validate(self.network,
                                                self.trldr, self.valdr,dict(self.train_profile),
                                                epoch,
                                                self.augm,startwith, self.loss_fn, self.opt,
                                                self.ustep,self.device)
        self.train_profile.update(tr_pro)
        return dict(self.train_profile)

    def predict(self, im, view_num_perscale=20):
        return predict(self.network, im, self.augm, view_num_perscale,
                       self.cls_num, self.device)

    def predict_with_scaleHist(self, im, tview=100, ):
        return predictby_scahist(self.network, im, self.augm, tview,
                                 self.cls_num, self.device)

    def load(self, ep):
        load_model(model=self.network, ep_num=ep)

        oltr_pr = load_train_hist(ep)
        self.train_profile.update(oltr_pr)

        augfr = load_train_hist(ep,name='augfr',)
        aug, si = self.augm
        aug.transforms[si].ScaleHist = np.array(augfr)


    def test(self):
        correct, total = 0, 0
        inds = self.teldr.dataset.indices
        dts = self.teldr.dataset.dataset.pathset
        for i in inds:
            obj = dts[i]
            img = dorg.imread(obj['im_path'])
            lb = dorg.ENCODE[obj['label']]
            out = self.predict_with_scaleHist(img,
                                              ).argmax(0).view(-1).item()
            correct += lb == out
            total += 1
            print(correct, total, correct / total)
        return correct / total
