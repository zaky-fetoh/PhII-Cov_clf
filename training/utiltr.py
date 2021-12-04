import torchvision.transforms as transf
import torch.nn.functional as f
from data_org import *
import torch.nn as nn
from model import *
import torch as t
import pickle


def save_train_hist(hist, ep_num,
                    name='hist',
                    outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.p'
    with open(file_name, 'wb') as file:
        pickle.dump(hist, file)


def load_train_hist(ep_num, name='hist',
                    outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.p'
    with open(file_name, 'rb')as file:
        hist = pickle.load(file)
    return hist


def accuracy(output, target, ):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        output = output.argmax(1)
        matchs = output == target
        return matchs.sum().float() / matchs.size(0)


def save_model(net, ep_num,
               name='weight',
               outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.pth'
    t.save(net.state_dict(), file_name)
    print('Model Saved', file_name)


def load_model(file_path=None, model=CNN(),
               outPath='./weights/', name='weight', ep_num=None,
               ):
    file_path = outPath + name + str(
        ep_num) + '.pth' if not file_path else file_path
    state_dict = t.load(file_path)
    model.load_state_dict(state_dict)
    print('Model loaded', file_path)


@t.no_grad()
def predict(net, im, augm=get_aug(),
            num_view=20, num_cls=2,
            device=t.device('cuda' if t.cuda.is_available() else 'cpu')):
    out = t.zeros(num_cls).to(device=device)

    aug, si = augm
    aug.transforms[si].frez_count()
    scale_len = aug.transforms[si].scales.__len__()

    for s in range(scale_len):
        aug.transforms[si].n = s
        ib = t.cat([aug(im).to(device=device).unsqueeze(0) for _ in range(num_view)],
                   dim=0)
        out += net(ib).mean(0)

    aug.transforms[si].unfrez_count()

    return out / scale_len


@t.no_grad()
def predictby_scahist(net, im, augm,
                      total_var=100, num_cls=2,
                      device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
                      ):
    aug, si = augm
    aug.transforms[si].frez_count()
    out = t.zeros(num_cls).to(device=device)

    for i, fr in enumerate(aug.transforms[si].ScaleHist):
        prop = fr / aug.transforms[si].total
        num_view = int(prop * total_var)
        aug.transforms[si].n = i
        if not num_view:
            continue
        ib = t.cat([aug(im).to(device=device).unsqueeze(0) for _ in range(num_view)],
                   dim=0)
        out += net(ib).mean(0) * (prop)

    aug.transforms[si].unfrez_count()

    return out


def train_(net, train_loader, criterion, opt_fn, augm, ustep,
           device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    aug, si = augm
    llis, alis, samples = list(), list(), 0
    aug.transforms[si].unfrez_count()
    for imgs, target in train_loader:
        aug.transforms[si].new_scale()
        imgs = imgs.to(device=device)
        target = target.to(device=device)
        samples += imgs.shape[0]

        pred = net(imgs)

        loss = criterion(pred, target)
        loss.backward()
        if samples >= ustep:
            print('netupdated:', samples)
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            opt_fn.step()
            opt_fn.zero_grad()
            samples = 0

        llis.append(loss.item())
        alis.append(accuracy(pred, target).item())

        print(llis[-1], alis[-1])

    return [llis, alis]


@t.no_grad()
def validate_(net, val_loader, criterion, augm,
              device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
              ):
    aug, si = augm
    llis, alis = list(), list()
    aug.transforms[si].frez_count()
    for imgs, target in val_loader:
        aug.transforms[si].new_scale()

        imgs = imgs.to(device=device)
        target = target.to(device=device)

        pred = net(imgs)
        loss = criterion(pred, target)

        llis.append(loss.item())
        alis.append(accuracy(pred, target).item())

        print(llis[-1], alis[-1])

    return [llis, alis]


def train_and_validate(net, trldr, valdr,
                       epochs=20, augm=None,
                       criterion=None, opt_fn=None, ustep = None,
                       device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
                       ):
    tr_profile = dict()
    aug, si = augm
    net.to(device=device)
    for e in range(epochs):
        tr_profile[e] = list()

        net.train()
        tr_profile[e] += train_(net, trldr, criterion,
                                opt_fn, augm, ustep,
                                device)

        save_model(net, e)

        net.eval()
        tr_profile[e] += validate_(net, valdr, criterion,
                                   augm, device)
        save_train_hist(tr_profile, e)
        save_train_hist(aug.transforms[si].ScaleHist.tolist(),
                        e, name='augfr')

    return tr_profile
