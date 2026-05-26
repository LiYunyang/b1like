import numpy as np

class model_base():

    def __init__(self, params=None):

        params_def = self.get_default_model_params()
        if params is None:
            print("here")
            params = params_def
        elif (len(params.keys()) < len(params_def.keys())):
            print("there")
            for key in params_def.keys():    
                if key not in params.keys():
                    params[key] = params_def[key]

        self.params = params

    def get_default_model_params(self):
        NotImplementedError('Call with experiment specific BPCM')
        return

    def model_spec(self):
        NotImplementedError('Call with experiment specific BPCM')
        return



class S4_model(model_base):
    def __init__(self,config,params=None, 
                 iscl=True, lmin=30):

        self.lencl = config.clout  # C_ells in K^2
        self.tencl = config.clten  # C_ells in K^2     

        self.iscl  = iscl
        self.lmin  = lmin

        #output in K^2 
        
        super().__init__(params=params)

    def get_default_model_params(self):
        #info from 
        #http://bicep.rc.fas.harvard.edu/CMB-S4/analysis_logbook/20200208_06_sims_details/
        # sync_freq in cmbs4/make_s4_gcomps.m , 
        # dust_freq, and fg_pivots are from defaults in multicomp/like_read_theory.m

        params = {}
        params['r'] = 0.0
        params['A_L'] = 1.0
        params['A_s'] = 3.8    #uK^2
        params['A_d'] = 4.25   #uK^2
        params['beta_s']  = -3.1
        params['alpha_s'] = -0.6
        params['beta_d']  = 1.6
        params['alpha_d'] = -0.4
        params['EB_s']    = 2.0
        params['EB_d']    = 2.0
        params['epsilon'] = 0.0
        params['T_d']     = 19.6 #K

        params['sync_freq'] = 23 #GHz 
        params['dust_freq'] = 353 #GHz
        params['fg_pivot']  = 80

        return params


    def model_spec(self, lmax=None):
        """       
        

          output are in K^2; Cl or l*(l+1)Cl/2pi
        """
        # 

        params = self.params

        camb_r = 0.01 #sky_yy/cmb/cls/ffp10_wtensors_params.ini

        if self.iscl:
            #cmb already in K^2 and Cl; fg in uK^2 and Dl
            lcmb = lambda l:1
            lfg  = lambda l:1.0/ (l*(l+1)/(2*np.pi) * 1e12) #to K^2
        else: 
            lcmb = lambda l:l*(l+1)/(2*np.pi) 
            lfg  = lambda l: 1.0/1e12 #to K^2
        
        ell = np.arange(lmax+1)
        ell[0] = 1 #to avoid nan/inf

        ms = {}
        for key in ['cmbl','cmbt','dust','sync']:
            ms[key]={}

        ms['cmbl']['TT'] = self.lencl['TT C_l'][:lmax+1]*lcmb(ell)
        ms['cmbl']['EE'] = self.lencl['E-mode C_l'][:lmax+1]*lcmb(ell)
        ms['cmbl']['BB'] = params['A_L']*(
                           self.lencl['B-mode C_l'][:lmax+1]*lcmb(ell))

        ms['cmbt']['BB'] = params['r'] / camb_r * (
                           self.tencl['B-mode C_l'][:lmax+1]*lcmb(ell))

        ms['dust']['BB'] = params['A_d']*(
                           ell/params['fg_pivot'])**params['alpha_d']*lfg(ell)
        ms['dust']['EE'] = ms['dust']['BB']*params['EB_d']

        ms['sync']['BB'] = params['A_s']*(
                           ell/params['fg_pivot'])**params['alpha_s']*lfg(ell)
        ms['sync']['EE'] = ms['sync']['BB']*params['EB_s']

        if self.lmin is not None: 
            for key in ['cmbl','cmbt','dust','sync']:
              for teb in ms[key].keys():
                ms[key][teb][:self.lmin+1]=0

        return ms


class BK_model(model_base):

    def __init__(self, params=None,
                 liketheorydir="/sdf/home/w/wlwu/bk/"):
    
        self.liketheorydir = liketheorydir
        super().__init__(params=params)

    def get_default_model_params(self):
        """
        r   =  tensor-to-scalar ratio
        A_L =  lensing amplitude (1 = standard LCDM lensing)
        A_s = synchrotron power spectrum amplitude, in
               uK_{CMB}^2, at specified freq and ell pivot 
        A_d = dust power spectrum amplitude, in uK_{CMB}^2, at_
              specified freq and ell pivot
        beta_s, synchrotron frequency spectral index
        alpha_s, synchrotron spatial spectral index
        beta_d, dust frequency spectral index
        alpha_d, dust spatial spectral index
        EB_s,  E/B ratio for synchrotron. 1 means that synchrotron 
              contributes equal amounts of E and B power; 2 means 
              that synchrotron contributes twice as much E as B;
        EB_d ratio for dust. 1 means that dust contributes
              equal amounts of E and B power; 2 means that dust
              contributes twice as much E as B; etc.
        epsilon, dust/sync correlation parameter
        T_d, dust temperature (19.6 K for PIP97 model)
        -----
        additional parameters put in likedata.opt through like_read_theory in BK
        sync_freq = freq for A_s  
        dust_freq = freq for A_d
        fg_pivot  = ell pivot for A_s and A_d

        """
        params = {}
        params['r'] = 0.0
        params['A_L'] = 1.0
        params['A_s'] = 0.0    #uK^2
        params['A_d'] = 3.75   #uK^2
        params['beta_s']  = -3.0
        params['alpha_s'] = -0.6
        params['beta_d']  = 1.6
        params['alpha_d'] = -0.4
        params['EB_s']    = 2.0
        params['EB_d']    = 2.0
        params['epsilon'] = 0.0
        params['T_d']     = 19.6 #K
        params['sync_freq'] = 23 #GHz 
        params['dust_freq'] = 353 #GHz
        params['fg_pivot']  = 80

        return params

    def model_spec(self, lmax=None):
        """ part of model_rms.m; leave n_exptfreq scaling to bpcm
        output
          model_spec: dictionary of spectra
              array of size(n_ell)
              n_ell = number of ells (unbinned)
          keys: cmbl, dust, sync, cmbt
          keys: TT, EE, BB
          output are in l*(l+1)Cl/2pi (dust/sync modeling in that)
        """
        import scipy.io as sio
        # BK CAMB input for making BK sims
        tmp = sio.loadmat(self.liketheorydir+"like_theory_bk14lt2894.mat")
        camb = tmp['liketheory'][0]
        
        #note that A_L and r only scales BB spectrum here

        params = self.params
        # ell starts at 1 (lmax=600)
        ell =  np.concatenate(camb['l'][0])
        assert lmax == ell[-1]

        ms = {}
        for key in ['cmbl','cmbt','dust','sync']:
            ms[key]={}

        ms['cmbl']['TT'] = camb['lcdm'][0][:,0]
        ms['cmbl']['EE'] = camb['lcdm'][0][:,1]
        ms['cmbl']['BB'] = params['A_L']*camb['lensing'][0][:,2]

        ms['cmbt']['BB'] = params['r']/camb['r'][0][0][0]*camb['tensor'][0][:,2]

        ms['dust']['BB'] = params['A_d']*(ell/params['fg_pivot'])**params['alpha_d']
        ms['dust']['EE'] = ms['dust']['BB']*params['EB_d']

        ms['sync']['BB'] = params['A_s']*(ell/params['fg_pivot'])**params['alpha_s']
        ms['sync']['EE'] = ms['sync']['BB']*params['EB_s']

        return ms
