# generate data files for cobaya; pipe b1
#
# run1: test49sims   np.cov() covmat, Cfl rescaled to match sim mean
# run1: test49sims1  bk-style covmat (wrong CMBxDust and DustxLT covariance; no type3 sims yet_
# run1: bpcmmask1    np.cov() covmat; with snmask
# run1: bpcmmask2    np.cov() w/ snmask; rescale noise (type6) and type8 DustxLT spectrum by the ratio of 
#                    np.sqrt(b1bpcm.rwf[1]*b1bpcm.rwf[2])/b1bpcm.rwf[4]
# run2: v1           bk-style covmat, with proper type3 sims for dust (7434x7461)
# run3: v1           same as run2 v1 but with better alpha/rho for LT (7434x7462)
#       v2           with nspec for dust+cmb not set to zero in all_nspec
# run4: v1           BK24 B3+Planck HFI with 7462 LT (has E-to-B because real BK24 TOD sims)
# run5: v1           BK23 B3+Planck HFI with 7462 LT (no E-to-B because scaling B3 noise)
# run5: v2           same as run5 v1; with nspec for dust+cmb not set of zero in all_nspec
# run6: v1           same as run3 v1 but with LTxLT type6 noise spec x2 smaller



import sys
sys.path.insert(1, "/sdf/home/w/wlwu/repos/compsep_code/configs/")
sys.path.insert(1, "/sdf/home/w/wlwu/repos/compsep_code/src/")
import os

import config_pipeb1_v1 as cf

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
run=3


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
 

#tmp -- "final" files
#aps93566 -- "arrays of aps" of different types
if run == 4: 
    bdir = rootdir+"bandpowers/7484x7462/"#7434x7461/"
    tmp = sio.loadmat(bdir+"xxxx_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0018_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_matrix_directbpwf.mat")
    aps93566 = sio.loadmat(bdir+"xxx1_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_matrix.mat")
if run==5:
    bdir = rootdir+"bandpowers/7434x7462/"
    tmp = sio.loadmat(bdir+"xxxx_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0018_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_matrix_directbpwf.mat")
    aps93566 = sio.loadmat(bdir+"xxx9_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2023_filtl3_weight4_gs_ring_dp1100_jack01_matrix.mat")
if run==6:
    bdir = rootdir+"bandpowers/7434x7463/" 
    tmp = sio.loadmat(bdir+"xxxx_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0018_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_matrix_directbpwf.mat")
    aps93566 = sio.loadmat(bdir+"xxx9_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0016_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_matrix.mat")
if run==3:
    bdir = rootdir+"bandpowers/7434x7462/" #or 7461
    tmp = sio.loadmat(bdir+"xxxx_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_xxxx_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_matrix_directbpwf.mat")
    aps93566 = sio.loadmat(bdir+"xxx9_B3_2018_filtl3_weight4_gs_ring_dp1100_jack01_0013_B3_2018_0015_B3_2018_0016_B3_2018_0016_B3_2018_matrix.mat")



#write Clhat, .dataset, covmat, bpwf
#start with just CMBxCMB, LTxLT, CMBxLT

#bandpowers
#tmp['r'][?][0][0] ?: 0-5; gives bin center
#tmp['r'][?][0][1] ?: 0-5; gives xspec type 'CMBxCMB', 'dustxdust', 'LTxLT', 'CMBxdust', ...
#tmp['r'][?][0][2-4] ?: 0-5; gives spectra (17, 6 or 9, 49); 17 bins, 49rlz, 6 (TT,TE,EE,BB,TB,EB); 9 if CMBxdust, CMBxLT, dustxLT; 2=type8 (cmb+dust+noise), 3=type5 (cmb) , 4=type6 (noise)
#

#input model
#tmp['inpmod'][0][0][1][:,i]; i goes through TT/TE/EE/BB/... CMB Dl spectrum
#tmp['inpmod'][0][0][2][:,i]; i goes through TT/TE/EE/BB/... CMB Cl spectrum


nmaps = 3
bins = np.arange(1,10)
nbins = len(bins)
nspec = 6
nsims = 49
bbidx=3

fname = "v2"
bkstyle=True

'''
In [789]: np.sqrt(b1bpcm.rwf[1]*b1bpcm.rwf[2])/b1bpcm.rwf[4]
Out[789]: 
array([0.00467116, 0.00468579, 0.01189884, 0.00993381, 0.0224575 ,
       0.03016506, 0.03638744, 0.04368846, 0.04880586])
'''
if run < 3:
    if fname in ["bpcmmask2", "v1"]:
        rwfratio = np.array([0.00467116, 0.00468579, 0.01189884, 0.00993381, 0.0224575 ,
           0.03016506, 0.03638744, 0.04368846, 0.04880586])
    else:
        rwfratio = np.ones_like([0.00467116, 0.00468579, 0.01189884, 0.00993381, 0.0224575 ,
           0.03016506, 0.03638744, 0.04368846, 0.04880586])
else: 
    rwf_raw = tmp['supfac']['rwf'][0]
    rwfratio = np.sqrt( rwf_raw[1]*rwf_raw[2])[bins,bbidx]/rwf_raw[5][bins, bbidx]
    print('rwfratio:')
    print(rwfratio)


# 11, 22, 33, 12, 23, 13
all_spec = np.concatenate([tmp['r'][0][0][2][bins,bbidx,:],
                          tmp['r'][1][0][2][bins,bbidx,:],
                          tmp['r'][2][0][2][bins,bbidx,:],
                          tmp['r'][3][0][2][bins,bbidx,:], #cmbxdust
                          tmp['r'][5][0][2][bins,bbidx,:] * rwfratio[:, None], #dustxLT
                          tmp['r'][4][0][2][bins,bbidx,:]], axis=0) #cmbxLT

all_nspec = np.concatenate([tmp['r'][0][0][4][bins,bbidx,:].mean(1),
                          tmp['r'][1][0][4][bins,bbidx,:].mean(1),
                          tmp['r'][2][0][4][bins,bbidx,:].mean(1),
                          #np.zeros_like( tmp['r'][3][0][4][bins,bbidx,:].mean(1) ), #cmbxdust !!! TURN BACK ON
                          tmp['r'][3][0][4][bins,bbidx,:].mean(1),
                          np.zeros_like( tmp['r'][5][0][4][bins,bbidx,:].mean(1) ), #dustxLT
                          np.zeros_like( tmp['r'][4][0][4][bins,bbidx,:].mean(1))] )

lmax_bpwf=600  #starts at ell=1 
#nell, totalbins, spec
#concatenate to start at ell=0
totbins = tmp['bpwf'][:,0][0][1].shape[1]
arr = np.array([0]*totbins)
all_bpwf = np.concatenate([np.concatenate([ arr[None,:], tmp['bpwf'][:,0][0][1][:lmax_bpwf, :, bbidx]]),
                           np.concatenate([ arr[None,:], tmp['bpwf'][:,1][0][1][:lmax_bpwf, :, bbidx]]),
                           np.concatenate([ arr[None,:], tmp['bpwf'][:,2][0][1][:lmax_bpwf, :, bbidx]]),
                           np.concatenate([ arr[None,:], tmp['bpwf'][:,3][0][1][:lmax_bpwf, :, bbidx]]),
                           np.concatenate([ arr[None,:], tmp['bpwf'][:,5][0][1][:lmax_bpwf, :, bbidx]]),
                           np.concatenate([ arr[None,:], tmp['bpwf'][:,4][0][1][:lmax_bpwf, :, bbidx]]),   
                           ], axis=1).transpose() #(nbin*nspec, nell)


b1bpcm = bpcm.BPCM_B1(cf, aps93566['aps'], all_bpwf, tmp['supfac']['rwf'],
                      bins,
                      n_sims=nsims,
                      params={'A_d':4.3, 'alpha_d':-0.4}, 
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

if 0: #compare np.cov() bpcm vs bk-stype
    #save BPCM
    #arrange all_spec so that it's in (nbin*nspec, nrlz) [collect nth bin entries together] 
    tspec = np.transpose(all_spec.reshape(nspec, nbins, nsims), (1,0,2))
    tspec = np.squeeze(tspec.reshape(1, nspec*nbins, nsims))

    N=nsims
    p=nbins*b1bpcm.n_spec  #use only 5 bins (usually) when sampling
    print(b1bpcm.n_spec)
    hartlap = (N-p-2.0)/(N-1)
    #bpcmout = 1/hartlap*np.cov(all_spec_v2_4cov, rowvar=False, ddof=1)
    bpcmout1 = np.cov(tspec, rowvar=True, ddof=1) #not hartlap correcting becuase not enough sims
    bpcmout1 = bpcmout1*bpcmmask

    #individual ss, nn, sn1-4 terms
    raw_bpcm = b1bpcm.raw_bpcm_terms()
    snmask = b1bpcm.construct_snmask(ncbands=b1bpcm.ncbands, scbands=b1bpcm.scbands)

    def cov2corr(cov):
        std_dev = np.sqrt(np.diag(cov))
        denom = np.outer(std_dev, std_dev)
        corr = cov / denom
        return corr

    figure(); imshow(cov2corr(bpcm_out)[:6,:6], cmap="RdBu_r", clim=[-1,1])
    figure(); imshow(cov2corr(bpcmout1)[:6,:6], cmap="RdBu_r", clim=[-1,1])

'''
In [286]: np.ones((6,6))*snmask['sig'][:6,:6]+np.ones((6,6))*snmask['noi'][:6,:6] + np.ones((6,6))*snma
     ...: sk['sn1'][:6,:6] + np.ones((6,6))*snmask['sn2'][:6,:6] + np.ones((6,6))*snmask['sn3'][:6,:6]
     ...: + np.ones((6,6))*snmask['sn4'][:6,:6]
Out[286]: 
array([[6., 1., 1., 3., 2., 3.],
       [1., 6., 0., 3., 0., 0.],
       [1., 0., 6., 0., 0., 3.],
       [3., 3., 0., 4., 2., 1.],
       [2., 0., 0., 2., 4., 2.],
       [3., 0., 3., 1., 2., 4.]])


'''


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

