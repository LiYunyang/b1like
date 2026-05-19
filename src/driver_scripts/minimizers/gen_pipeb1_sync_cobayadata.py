# same as gen_pipeb1_cobayadata.py but now read in 
# final files and aps966-type files knowing that they 
# also have sync in the auto/cross-spec

import sys
sys.path.insert(1, "/sdf/home/w/wlwu/repos/compsep_code/configs/")
sys.path.insert(1, "/sdf/home/w/wlwu/repos/compsep_code/src/")
import os

import config_pipeb1_v2 as cf
import mutils

from sampling import bpcm
from sampling import model 
from sampling import write_outputs as wo

import numpy as np
import scipy.io as sio
import pickle as pk
import subprocess
from pathlib import Path
import argparse

endidx = 49
run=7

rootdir = "/sdf/home/w/wlwu/data/pipeb1/"
#spec file loc
#bandpass location
auxdir = rootdir+"" #TBD may not need bandpass if don't care about accurate dust fit params

ilc_weight_fname = None #need if want to model/marginalized over residuals from statistical beta_d variation

#output cobaya file dir
odir="/sdf/home/w/wlwu/data/pipeb1/sampler_io/run%i/data/"%(run)
Path(odir).mkdir(parents=True, exist_ok=True)
Path(odir+"windows/").mkdir(parents=True, exist_ok=True)
odir1="/sdf/home/w/wlwu/data/pipeb1/sampler_io/run%i/"%(run)
Path(odir1+"batchfiles/").mkdir(parents=True, exist_ok=True)
Path(odir1+"farm/").mkdir(parents=True, exist_ok=True)
Path(odir1+"outputs/").mkdir(parents=True, exist_ok=True)
Path(odir1+"yaml_in/").mkdir(parents=True, exist_ok=True)
os.chdir(odir)

bands = None #B3, Planck HFI (check with dom what the input maps are; needed if modeling residuals from dust

#final -- "final" files (called "tmp" in gen_pipeb1_cobayadata.py)
#aps966 -- "arrays of aps" of different simtypes

if run==7:
    bdir = rootdir+"bandpowers/7435x7462/"
    final = sio.loadmat(bdir+"xxxx_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0018_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_matrix_directbpwf.mat")
    aps966 = sio.loadmat(bdir+"xxx9_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_matrix.mat")

nmaps = len(cf.maporder)  
nspec = int(nmaps*(nmaps+1)/2)
bins = np.arange(1,nspec)
nbins = len(bins)
nsims = 49
bbidx=3

fname = "v1"
bkstyle = True


#rwfratio for DUSTxLT and SYNCxLT 
#they are needed because LT is debiased in the final file in a funny way
rwf_raw = final['supfac']['rwf'][0]
rwfratios_arr = []

for i in range(0, nmaps):
    idx_x = mutils.input_ordering_index( i, i, nmaps )
    rwfratios_arr.append( mutils.get_rwfratio(rwf_raw, i, i, idx_x, nmaps)[bins, bbidx] ) 

for diff in range(1, nmaps):
    for i in range(nmaps - diff):
        j=i+diff
        idx_x = mutils.input_ordering_index( i, j, nmaps )
        rwfratios = mutils.get_rwfratio(rwf_raw, i, j, idx_x, nmaps)[bins, bbidx]
        rwfratios_arr.append( rwfratios )
        if j==nmaps-1:
            print(i,j,idx_x)
            print(rwfratios)

#rwfratio = np.sqrt( rwf_raw[1]*rwf_raw[2])[bins,bbidx]/rwf_raw[5][bins, bbidx]
#print('rwfratio:')
#print(rwfratio)

mapping, rev_mapping, ori_pair2idx, tar_pair2idx, ori_idx2pair, tar_idx2pair  = mutils.reorder_pairs(nmaps)
# from 11 22 33 44 12 23 34 13 24 14
# to   11 22 33 44 12 13 14 23 24 34

all_spec  = np.concatenate([ final['r'][mapping[ii]][0][2][bins, bbidx,:] * rwfratios_arr[ii][:, None] 
                            for ii in range(nspec) ], axis=0) 

#don't need rwfratios because the dustxLT and syncxLT noise spec are zero
all_nspec = np.concatenate([ final['r'][mapping[ii]][0][4][bins,bbidx,:].mean(1)
                             if not (nmaps-1 in tar_idx2pair[ii] and 
                                 tar_idx2pair[ii][0] != tar_idx2pair[ii][1]) else np.zeros_like( final['r'][0][0][4][bins,bbidx,:].mean(1) ) 
                             for ii in range(nspec) ])

lmax_bpwf=600  #starts at ell=1 
#nell, totalbins, spec
#concatenate to start at ell=0
totbins = final['bpwf'][:,0][0][1].shape[1]
arr = np.array([0]*totbins)
all_bpwf = np.concatenate([ np.concatenate([ arr[None,:], final['bpwf'][:, mapping[ii]][0][1][:lmax_bpwf, :, bbidx]]) 
                            for ii in range(nspec) ], axis=1).transpose()


b1bpcm = bpcm.BPCM_B1(cf, aps966['aps'], all_bpwf, final['supfac']['rwf'],
                      bins,
                      n_sims=nsims,
                      params={'A_d':4.3, 'alpha_d':-0.4, 'A_s':1e-8, 'alpha_s':-0.6},
                        )

bpcmmask = b1bpcm.construct_ell_mask()


if bkstyle:
    #construct BK-style BPCM
    bpcm_out = b1bpcm.scale_bpcm(no_scale=True)
    bpcmout = bpcm_out*bpcmmask
else: #np.cov
    #arrange all_spec so that it's in (nbin*nspec, nrlz) [collect nth bin entries together]
    tspec = np.transpose(all_spec.reshape(nspec, nbins, nsims), (1,0,2))
    tspec = np.squeeze(tspec.reshape(1, nspec*nbins, nsims))

    snmask = b1bpcm.construct_snmask(ncbands=b1bpcm.ncbands, scbands=b1bpcm.scbands)
    sntotmask = np.zeros_like(snmask['sig'])
    for key in snmask.keys():
        sntotmask += snmask[key]
    sntotmask = np.clip(sntotmask, 0, 1)

    bpcmout = np.cov(tspec, rowvar=True, ddof=1) #not hartlap correcting becuase not enough sims
    bpcmout = bpcmout*bpcmmask*sntotmask

if 1: #write files
    wo.bpwf(b1bpcm, fname)
    wo.bpcm(cf, b1bpcm, fname, bpcmout)
    Cfl_tmp = wo.Cfl_b1(cf, b1bpcm, fname, nmaps) #, scale_cfl=1.0) #check LLCDM, LT, LTxLCDM expectation
    #calibrate Cfl to match mean sims that enter BPCM
    ratio = Cfl_tmp / all_spec.mean(1).reshape(nspec, nbins).transpose()
    print(ratio)

    #replace _fiducial.dat file
    if 0:
        bpstr = wo.get_bpstr(cf,b1bpcm)
        f=open(fname+'_fiducial.dat','wt')
        f.write('#%s \n'%bpstr)
        for i in range(b1bpcm.nbins):
           #first column denotes n-th bin used in analysis
           tmp = "".join([' {0:0.6g}'.format(cl) for cl in np.nan_to_num(Cfl_tmp[i,:]/ratio[i,:])])
           f.write( "%i %s\n"%(i+1, tmp))
        f.close()

    #save Clhat
    for ii, idx in enumerate(np.arange(1,50)):

        fname_simn = fname+"_sim%04d"%idx
        clhat = all_spec[:,ii].reshape(nspec, nbins)
        wo.Clhat(cf, b1bpcm, fname_simn, clhat)
        wo.dataset(cf, b1bpcm, fname, fname_simn)


    #save Nl
    wo.Nl(cf, b1bpcm, fname, all_nspec.reshape(nspec,nbins))

    wo.params(fname)
    wo.bandpass(b1bpcm)
    










