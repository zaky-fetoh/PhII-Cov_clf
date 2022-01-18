import model as mdl
import training as tr
import data_org as dorg


"""
training a CNN with single level, 10, SPP layer using averagpooling 
as aggregating function
traing schadule fir40ep -> 1e3
       schadule sec50ep -> 1e4
       schadule thr40ep -> 1e5
"""

if __name__ == '__main__':
    pres = 80
    mdl.t.cuda.empty_cache()
    mobj = tr.Network(bs=16,ustep=256,cfx=mdl.saspp)
    mobj.opt = tr.optim.Adam(mobj.network.parameters(),1e-3)
    #mobj.load(pres)
    mobj.fit(epoch=40,)#startwith=pres+1)
    #mobj.test()
