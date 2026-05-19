
import numpy as np
import pickle as pk

import warnings

# Local
from cobaya.likelihoods.base_classes import CMBlikes
from cobaya.conventions import Const

#use bicep_keck_2015 functions
#from cobaya.likelihoods.bicep_keck_2015 import bicep_keck_2015 as bk15

#from camb.mathutils import threej_coupling
import warnings

Ghz_Kelvin = Const.h_J_s / Const.kB_J_K * 1e9

class b1like(CMBlikes):

    def get_requirements(self):
        req = super().get_requirements()
    
        if self.llcdm_band is not None:
            req["Cl_lensed_scalar"] = req["Cl"]
        return req

    def init_params(self, ini):
        #same as bicep_keck_2015
        super().init_params(ini)

        #self.fpivot_dust = ini.float('fpivot_dust', 353.0) #not used
        self.lpivot      = 80

        print('pcl_lmin: %i'%(self.pcl_lmin))
        print('pcl_lmax: %i'%(self.pcl_lmax))


        self.llcdm_band = None
        self.dust_band = None
        self.sync_band = None
        for i, mapname in enumerate(self.used_map_order):
            #this works for as ILCxLT only; 
            assert("LT" not in self.used_map_order[0])
            print(mapname)
            if ('LT' in mapname):  #need to make sure it's not DustxLT 
                self.llcdm_band = i
                print("adding llcdm_band")
            if 'Dust' in mapname:
                self.dust_band = i
            if 'Sync' in mapname:
                self.sync_band = i

    def get_dust_spec(self, data_params):
        '''
        returns dust spec starting at pcl_lmin up to pcl_lmax
        dust map returned at 353Ghz (delta function bandpass)
        '''
        Adust = data_params['BBdust']
        alphadust = data_params['BBalphadust']

        ells = np.arange(self.pcl_lmax+1)
        ells[0] = 1

        ratio    = ells / self.lpivot
        dustspec = Adust * ratio ** alphadust

        return  dustspec

    def get_sync_spec(self, data_params):
        '''
        returns sync spec starting at pcl_lmin up to pcl_lmax
        sync map returned at 23Ghz(?) (delta function bandpass)
        '''
        Async = data_params['BBsync']
        alphasync = data_params['BBalphasync']

        ells = np.arange(self.pcl_lmax+1)
        ells[0] = 1

        ratio    = ells / self.lpivot
        syncspec = Async * ratio ** alphasync

        return syncspec

    def get_theory_map_cls(self, Cls, data_params=None, lensed_scalar=None):
        #print(data_params)
        #print('aa:' , lensed_scalar)
        #print('bb:' , self.llcdm_band)
        if self.llcdm_band is not None: assert(lensed_scalar is not None)
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                this_cl = Cls.copy() #total Cls
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = this_cl.get(combination) 
                # DustxDust
                if i == self.dust_band and j == self.dust_band:
                    #print("getting dust spec")
                    cls = self.get_dust_spec(data_params)
                # SyncxSync
                elif i == self.sync_band and j == self.sync_band:
                    #print("getting sync spec")
                    cls = self.get_dust_spec(data_params)
                # Dust/SyncxCMB and Dust/SyncxLT
                elif (i == self.dust_band or j == self.dust_band or
                      i == self.sync_band or j == self.sync_band): 
                    cls = None
                # CMBxLT and LTxLT
                elif i == self.llcdm_band or j == self.llcdm_band:
                    #print("getting lensed scalar")
                    this_cl = lensed_scalar.copy()
                    cls = this_cl.get(combination)

                if cls is not None:
                    tmp = cls[self.pcl_lmin:self.pcl_lmax + 1]
                    CL.CL[:] = tmp
                else:
                    CL.CL[:] = 0
        #import ipdb
        #ipdb.set_trace()
        #print(self.map_cls[1,1].CL)
        self.adapt_theory_for_maps(self.map_cls, data_params or {})
        #add_foregrounds called in here; only add to CL.CL[0,0]
        #which needs to be the ILC band
        #print(self.map_cls[0,0].CL)



    def logp(self, **data_params):
        #print("in logp")
        cls = self.provider.get_Cl(ell_factor=True)
        lensed_scalar_cls = None
        if self.llcdm_band is not None:
            #print("get lensed_scalar")
            lensed_scalar_cls = self.provider.get_Cl_lensed_scalar(ell_factor=True)
            #lensed_scalar_cls = self.provider.get_lensed_scal_Cl(ell_factor=True)
            #scale the BB lensing spec
            #Al = data_params["Al_scale"]
            #lensed_scalar_cls['bb'] *= Al
        return self.log_likelihood(cls,
                      lensed_scalar=lensed_scalar_cls, **data_params)

    def transform_alt(self, C, Chat, Cfhalf):
        '''
        C: signal+noise model expectation
        Chat: data bandpowers
        Cfhalf: matrix_sqrt(C_fl)

        '''
        import scipy as sp
        Cinvhalf = np.linalg.inv(sp.linalg.sqrtm(C))
        diag, U = np.linalg.eigh( Cinvhalf @ Chat @ Cinvhalf )
        assert 0    


    def log_likelihood(self, dls, lensed_scalar=None, **data_params):
        r"""
        Get log likelihood from the dls (CMB C_l scaled by L(L+1)/2\pi)

        :param dls: dictionary of d_l ('tt', etc)
        :param data_params: likelihood nuisance parameters
        :return: log likelihood
        """
        self.get_theory_map_cls(dls, data_params, lensed_scalar=lensed_scalar,
                                )
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(
                    C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                  with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    self.transform(
                        C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                    #import ipdb
                    #ipdb.set_trace()
                    if len(w) > 0:
                        if w[-1].category==RuntimeWarning:
                           print("sqrt issue")
                           return -np.inf
                except np.linalg.LinAlgError:
                    self.log.debug("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used:(b + 1) * self.ncl_used] = vecp[
                self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self._fast_chi_squared(self.covinv, big_x)







