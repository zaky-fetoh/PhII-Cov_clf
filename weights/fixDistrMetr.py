from glob import glob
import pickle

def save_train_hist(hist, ep_num,
                    name='hist',
                    outPath='./weights/'):
    file_name = outPath + name + str(ep_num) + '.p'
    with open(file_name, 'wb') as file:
        pickle.dump(hist, file)


def fix():
    accum = dict()
    for file_name in glob('hist*'):
        with open(file_name, 'rb')as file:
            hist = pickle.load(file)
        accum.update(hist)
    return accum



def load_train_hist(file_name):
    with open(file_name, 'rb')as file:
        hist = pickle.load(file)
    return hist
