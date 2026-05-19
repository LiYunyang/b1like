# test49sims : np.cov() no snmask
# test49sims1: bkstyle cov (with snmask)
# bpcmmask1: np.cov() with snmask (for CMB+Dust runs becuase no 566 for type3(dust))
#
# need to have imported cobaya_theory_b1 in ../../src/b1like first


import yaml
import numpy as np
import os
from cobaya.yaml import yaml_load_file

#cases to run
# CMB
# CMB,LT
# CMB,Dust
# CMB,Dust,LT
# CMB,Dust,Sync
# CMB,Dust,Sync,LT
# changes fname



sets = [
    #"CMB",
    #"CMB,LT",
    "CMB,Dust",
    "CMB,Dust,LT",
    "CMB,Dust,Sync",
    "CMB,Dust,Sync,LT"
]

freeal = False
run = 7

datfname = "v1"#"test49sims1" #name of yaml_in yamlfile
ntry  = 1

idxarr = np.arange(1,50)

bdir = "/sdf/home/w/wlwu/data/pipeb1/sampler_io/run%i/"%run
outdir = bdir+"outputs/"
yamldir = bdir+"yaml_in/"
os.makedirs(bdir, exist_ok=True)
os.makedirs(yamldir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)

for set1 in sets:
    comps = set1.split(',')
    #comps = ['CMB', 'Dust', 'LT']
    fname_suf = ''.join([comp[0].lower() for comp in comps])
    outfname = '%s%s_%i'%(fname_suf, datfname, ntry)

    info = yaml_load_file("yaml_base_b1.yaml")

    print(comps)

    info["likelihood"]["b1like"]["dataset_params"]["maps_use"] = ["%s_B"%comp for comp in comps]

    for idx in idxarr: 
        print(idx)

        info["likelihood"]["b1like"]["dataset_file"]="%sdata/%s_sim%04d.dataset"%(bdir,datfname,idx)

        if 'Dust' not in comps:
            #fix dust params
            info['params']['BBalphadust']=-0.4
            info['params']['BBdust']=4.3
        if 'Sync' not in comps:
            #fix dust params
            info['params']['BBalphasync']=-0.6
            info['params']['BBsync']=0.0

        if freeal:
            info['params']['Al_scale'] = {'prior': {'dist':'uniform', 'max':2, 'min':0},
                                          'proposal': 1,
                                          'ref': {'dist':'norm', 'loc':1, 'scale':0.3}
                                         }

        info["output"]="%s%s_al%i_sim%04d"%(outdir, outfname, freeal, idx)

        yaml.dump(info, open("%s%s_al%i_sim%04d.yaml"%(yamldir, outfname, freeal, idx), 'w'))

