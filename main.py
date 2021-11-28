import model as mdl
import training as tr
import data_org as dorg


"""
Hello I'm there
test
"""

if __name__ == '__main__':
    mdl.t.cuda.empty_cache()
    mobj = tr.Network(bs=128,)

    thist = mobj.test()
