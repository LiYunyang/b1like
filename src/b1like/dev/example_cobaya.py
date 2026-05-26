import healpy as hp
import numpy as np
from cobaya.run import run
from cobaya.model import get_model

def rebincl(ell,cl, Nbins, minell, maxell):
    ''' Simple code to rebin the spectrum '''
    bb   = np.linspace(minell,maxell,Nbins+1)
    ll   = (bb[:-1]).astype(np.int_)
    uu   = (bb[1:]).astype(np.int_)
    ret  = np.zeros(Nbins)
    retl = np.zeros(Nbins)

    for i in range(0,Nbins):
        ret[i]  = np.mean(cl[ll[i]:uu[i]])
        retl[i] = np.mean(ell[ll[i]:uu[i]])

    return retl,ret

###### experimental setup ######
fsky  = 0.06             
fwhm  = 1.2          # [arcmin]
nlevt = 10           # [uk-arcmin]
nbins = 20
lmin  = 100
lmax  = 2000
################################

# usually want to compute to slightly higher ell
lmaxs    = lmax + 250 
ll       = np.arange(lmaxs+1)


# compute binning
bine     = np.linspace(lmin,lmax,nbins+1)
bins     = (bine[1:]+bine[:-1])/2.0
dl       = bine[1:]-bine[:-1]


# generate simulated data vector at a given comsology
fid_pars = {
            'omch2': 0.115 , 
            'ombh2': 0.022 , 
            'H0'   : 68    , 
            'tau'  : 0.06  ,
            'ns'   : 0.96  , 
            'As'   : 2.1e-9,
            'mnu'  : 0.00  ,
            'nnu'  : 3.046 ,
           }

info_fid = {
            'params'    : fid_pars,
            'likelihood': {'one': None},
            'theory'    : {'camb': {"extra_args": {"num_massive_neutrinos": 0}}},
           }

model_fid = get_model(info_fid)

model_fid.add_requirements({"Cl": {'tt': lmaxs}})
model_fid.logposterior({})
cls       = model_fid.provider.get_Cl(ell_factor=True, units="muK2")
cltt      = cls['tt'][:lmaxs + 1]
rl,dvec   = rebincl(ll,cltt,nbins,lmin,lmax)


# generate noise spectrum
bl        = hp.gauss_beam(fwhm=fwhm*0.000290888,lmax=lmaxs)   
nltt      = np.ones(lmaxs+1)*(np.pi/180./60.*nlevt)**2/bl**2 
nltt      = nltt*ll*(ll+1)/2./np.pi


# compute analytical covariance
rl,dltt11 = rebincl(ll,cltt[:lmax+1]+nltt[:lmax+1],nbins,lmin,lmax)
rl,dltt22 = rebincl(ll,cltt[:lmax+1]+nltt[:lmax+1],nbins,lmin,lmax)
rl,dltt12 = rebincl(ll,cltt[:lmax+1],nbins,lmin,lmax)
tmpTTTT   = 1/(2*bins+1)/dl/fsky*(dltt12*dltt12+dltt11*dltt22);
covTTTT   = np.zeros((nbins,nbins));
covTTTT[np.diag_indices(nbins)] = tmpTTTT
icov     = np.linalg.pinv(covTTTT)

# Compute loglikelihood
def simple_like(_self=None):
  # extract lensed spectra at each point
  cltt    = _self.provider.get_Cl(ell_factor=True, units="muK2")['tt'][:lmaxs+1]
  # rebin cls
  rl,tvec = rebincl(ll,cltt,nbins,lmin,lmax)
  X       = dvec - tvec
  logl    = -0.5*np.dot(X,np.dot(icov,X))
  return logl

# Parameters to sample 
pars  = {
        'omch2': {'prior': {'min': 0.10 , 'max': 0.13}   , 'latex': r'\Omega_{\rm c}h^{2}'},
        'ombh2': {'prior': {'min': 0.021, 'max': 0.023} , 'latex': r'\Omega_{\rm b}h^{2}'},
        'H0'   : {'prior': {'min': 67   , 'max': 69}     , 'latex': r'H_{0}'},
        'ns'   : {'prior': {'min': 0.95 , 'max': 0.97}  , 'latex': r'n_{\rm s}'},
        'As'   : {'prior': {'min':2e-09 , 'max': 2.2e-09}, 'latex': r'A_{\rm s}'},
        'tau'  : 0.06,
        'mnu'  : 0.00,
        'nnu'  : 3.046,
        }

# main settings for the run
info =  {
         #parameters to use: in this case we are using the parameters listed above
         "params"    : pars,

         # likelihood. In this case, using the simple_like listed above.
         # requires defines all the output from CAMB that is required. 
         "likelihood": {'testlike': {
                                     "external": simple_like,
                                     "requires": {'Cl': {'tt': lmaxs,'te': lmaxs, 'ee': lmaxs , 'pp': lmaxs },},
                                    }},
         # Which sampler to use. default mcmc is using Metropolis Hasting 
         "sampler"   : {"mcmc": {"max_samples": 1000000, "Rminus1_stop": 0.01, "max_tries": 100000}},

         # Which Boltzmann code to use to get theory. In this case we are using CAMB. Can switch to CLASS.
         "theory"    : {"camb": {"extra_args": {#"accurate_massive_neutrino_transfers": True,
                                               #"redshifts": [0.],
                                               "nonlinear": 'both',
                                               #"kmax": 50.,
                                               #"max_eta_k":20000,
                                               "num_massive_neutrinos"  : 0,
                                               #"halofit_version": 'takahashi',
                                               "dark_energy_model": "ppf",
                                               #"WantTransfer": True
                                               },
                                'stop_at_error': True
                                }
                        },

         # This is the root of the files to output
         "output"    : 'chain_test',

         # Turn this to "True" to overwrite if chainfile already exists
         "force"     : True
        }


# Run chain
updated_info, products = run(info)


