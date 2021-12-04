import model as mdl
import training as tr
import data_org as dorg


"""
Hello I'm there
test
"""

if __name__ == '__main__':
    mdl.t.cuda.empty_cache()
    mobj = tr.Network(bs=32,ustep=256,)
    mobj.opt = tr.optim.Adam(mobj.network.parameters(),1e-4)
    mobj.load(13)
    mobj.fit(startwith=14)
    #thist = mobj.test()
