import model as mdl
import training as tr
import data_org as dorg


"""
Hello I'm there
test
"""

if __name__ == '__main__':
    pres = 22
    mdl.t.cuda.empty_cache()
    mobj = tr.Network(bs=16,ustep=256,)
    mobj.opt = tr.optim.Adam(mobj.network.parameters(),1e-2)
    # mobj.load(pres)
    mobj.fit(epoch=30,startwith=0)
    #mobj.test()
