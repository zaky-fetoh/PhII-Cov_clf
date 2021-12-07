import model as mdl
import training as tr
import data_org as dorg


"""
Hello I'm there
test
"""

if __name__ == '__main__':
    pres = 30
    mdl.t.cuda.empty_cache()
    mobj = tr.Network(bs=16,ustep=256,cfx=mdl.SameConv2d)
    mobj.opt = tr.optim.Adam(mobj.network.parameters(),1e-4)
    mobj.load(pres)
    mobj.fit(epoch=50,startwith=pres+1)
    #mobj.test()
