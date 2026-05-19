#follow example in /sdf/home/w/wlwu/repos/2025/202503_cosmopower/cobaya_theory_cpj.py

from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict

import os, sys
import numpy as np

def get_BKsim_spec(fname="/sdf/home/w/wlwu/repos/compsep_code/src/b1like/fixlcdm_spec.npz"):
    '''
    the lensing BB spec should match the spt-3g sims 
    use the BK generated tensor r=1 spec to scale for the tensor BB

    '''
    liketheorydir="/sdf/home/w/wlwu/bk/"

    if not os.path.isfile(fname):

        import scipy.io as sio
        # BK CAMB input for making BK sims
        tmp = sio.loadmat(liketheorydir+"like_theory_bk14lt2894.mat")
        camb = tmp['liketheory'][0]
        r=camb['r'][0][0][0]

        # ell starts at 1 (lmax=600); in Dl [uK^2]
        dltens = np.concatenate([[1e-10], camb['tensor'][0][:,2]/r]) #now starts at ell=0

        #SPT camb; in Dl, starts at ell=2
        tmp1 = "/sdf/home/w/wlwu/repos/spt3g_software/simulations/data/camb/planck18_TTEEEE_lowl_lowE_lensing_highacc/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat"
        ell,sltt,slee,slbb,slte=np.loadtxt(tmp1,unpack=True)

        dlbb = np.concatenate([slbb[0:2], slbb]) #now starts at ell=0

        np.savez(fname, cmbt=dltens, cmbl=dlbb)

    return np.load(fname)  #tmp['cmbt'] gives r=1 bb spec; tmp['cmbl'] gives lensing bb spec


class fix_lcdm(BoltzmannBase):
    """
    class with fixed lensing BB spectrum and r=0.1(?) spectrum
    that can be scaled with Al_scale and r parameters.
    The fixed spectra comes from the input to generating the BK sims

    (the llcdm spec should match with what's input to spt-3G)

    """
    fixlcdm_fname="/sdf/home/w/wlwu/repos/compsep_code/src/b1like/fixlcdm_spec.npz"
    
    extra_args: InfoDict = { }

    def initialize(self):
        super().initialize()

        #have all objects, when relevant, start at ell=1 (bpwf, theory spec) 

        spec = get_BKsim_spec(self.fixlcdm_fname)
        tens_bb = spec['cmbt']
        lens_bb = spec['cmbl']

        self.lmax_theory = 999 #self.extra_args.get("lmax_theory", 999)

        self.tens_bb = np.concatenate([tens_bb, np.zeros(self.lmax_theory-len(tens_bb)+1)])
        self.lens_bb = lens_bb[:self.lmax_theory+1]

        self.r_parname  = "r"
        self.Al_parname = "Al_scale"

        ells = np.arange(0, self.lmax_theory+1)
        ells[0] = 1 #avoid divide by zero
        self.ells = ells

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """

        self.provider = provider
    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {self.r_parname: None, self.Al_parname: None}

    def must_provide(self, **requirements):
        """
        Return dictionary of parameters that must be provided.
        """
        return {self.r_parname: None, self.Al_parname: None}

    def get_can_provide(self):
        """
        Return list of quantities that can be provided.
        """
        return ["Dl" ]

    def calculate(self, state, want_derived = True, **params):
        """
        Return total Cl/Dl, lensed_scalar Cl/Dl

        """


        ellfac = self.ells * (self.ells+1) / 2 / np.pi

        state["Dl"]={
                "ell": self.ells,
                "bb" : state['params'][self.r_parname] * self.tens_bb + state['params'][self.Al_parname] * self.lens_bb}

        state["Cl"]={"ell": self.ells, 
                    "bb": state["Dl"]["bb"]/ellfac}

        state["Dl_lens"] = { "ell": self.ells,
                             "bb": state['params'][self.Al_parname] * self.lens_bb }

        state["Cl_lens"] = { "ell": self.ells,
                             "bb": state["Dl_lens"]["bb"] / ellfac }

        return

    def get_Cl(self, ell_factor=False, **kwargs):
        """
        Return the total (tensor+lensed_scalar) Cls or Dls
        """
        if ell_factor:
            return self.current_state["Dl"].copy()
        else:
            return self.current_state["Cl"].copy()

    def get_Cl_lensed_scalar(self, ell_factor=False, **kwargs):
        """
        Return the lensed_scalar Cls or Dls
        """
        if ell_factor:
            return self.current_state["Dl_lens"].copy()
        else:
            return self.current_state["Cl_lens"].copy()

