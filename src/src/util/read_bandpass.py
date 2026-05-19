import numpy as np
import scipy.io as sio
import astropy.io.fits as afits


def read_bandpass(exptfreqs, bpdir):
    """
    input:
      exptfreq: list(str) 
         experiment name
      bpdir: dictionary[B/K/L/P/w] for directory to read bandpasses

    output:
      bandpass: dictionary[exptfreq] that gives bandpasses
                  bp[0]: freq GHz
                  bp[1]: transmission
    """

    bpfname = expt_bandpass_dict()

    bandpass = {}

    for ef in exptfreqs:
        if ef == 'LT': 
          bandpass[ef] = None 
        else: 
          firstl = bpfname[ef][0]
          thisdir = bpdir[firstl]
          if firstl=='B' or firstl=='K':
              bandpass[ef] = read_BK_bandpass(thisdir+bpfname[ef])
          elif firstl=='L':
              bandpass[ef] = read_LFI_bandpass(thisdir+bpfname[ef])
          elif firstl=='P':
              bandpass[ef] = read_Planck_bandpass(thisdir+bpfname[ef])
          elif firstl=='w':
              bandpass[ef] = read_WMAP_bandpass(thisdir+bpfname[ef])
          else:
              assert(0)

    return bandpass

def get_dnu(nu):
    """
    input:
      nu: array of potentially unevenly spaced freq 
    output:
      dnu: freq steps
    """
    N = len(nu)

    dnu = np.zeros(N)
    dnu[0]    = nu[1]-nu[0]
    dnu[1:-1] = (nu[2:]-nu[0:-2])/2.0
    dnu[-1]   = nu[-1] - nu[-2]

    return dnu

def get_bandcenter(bp):
    """
    inputs:
        bp: array
          bp[0] frequency GHz
          bp[1] transmission
    output: 
        bandcenter: float
    """
    dnu = get_dnu(bp[0])

    bc = np.sum(dnu * bp[0] * bp[1])/np.sum(dnu * bp[1])  

    return bc

def read_BK_bandpass(fname):
    """nu: bp[0]; bandpass: bp[1]"""   
    bp = np.loadtxt(fname, delimiter=",", usecols=[0,1]).transpose() 

    return bp

def read_Planck_bandpass(fname):
    """nu: bp[0]; bandpass: bp[1]"""   

    hdu = afits.open(fname)
    #take info between 10-1000GHz
    i1 = np.where(hdu[1].data['frequency'] >= 10)[0][0]
    i2 = np.where(hdu[1].data['frequency'] <= 1000)[0][-1]
    bp = np.array([hdu[1].data['frequency'][i1:i2+1],
                    hdu[1].data['transmission'][i1:i2+1]]).astype(np.float64)
    return bp
    

def read_LFI_bandpass(fname):
    """nu: bp[0]; bandpass: bp[1]"""   

    bp = np.loadtxt(fname, usecols=[0,1])

    return bp.transpose()

def read_WMAP_bandpass(fname):
    
    #number of differencing assemblies
    nda = int(fname[-1])
  
    data_all = []
    for i in range(nda):
        for jj in range(2):
            fnamef = fname[:-1]+'%i%i_v5.cbp'%(i+1,jj+1)

            tmp = np.loadtxt(fnamef)
            data_all.append(tmp[:,1])
            data_all.append(tmp[:,2])
    
    #return array[0] --> nu; array[1] --> bp
    #Average across all differencies assemblies, both amplifiers.
    nu = tmp[:,0]
    bpmean = np.mean(data_all, axis=0)
    #Switch from RJ to spectral radiance convention.
    bpmean /= nu**2
    bpmean /= np.max(bpmean)
    bp = np.array([nu, bpmean]) 

    return bp

def expt_bandpass_dict():
    """
    return dictionary of exptfreq to bandpass file name
    """

    bpfname = {}
    bpfname.update(dict.fromkeys(['BK14_150','BK15_150','BK18_150','BK23_150'], 
                   "B2_frequency_spectrum_20141216.txt"))
    bpfname.update(dict.fromkeys(['BK14_95','BK15_95','BK18_K95','BK23_95'], 
                   "K95_frequency_spectrum_20150309.txt"))
    bpfname.update(dict.fromkeys(['B3_95','BK18_B95', 'B18_95', 'BK18_B95ext'], 
                   "B3_frequency_spectrum_20160101.txt"))
    bpfname.update(dict.fromkeys(['K210','BK18_210'], 
                   "K210_frequency_spectrum_20160101.txt"))
    bpfname.update(dict.fromkeys(['BK15_220','BK18_220','BK23_220'], 
                   "K220_frequency_spectrum_20160120.txt"))

    for freq in [30,44,70]:
        bpfname['P0%i'%freq] = 'LFI_BANDPASS_F0%i.txt'%freq
    bpfname['P100'] = 'Planck_Bandpass_100GHz_DX11d_FULL.fits'
    for freq in [143,217,353]:
        bpfname['P%i'%freq] = 'Planck_Bandpass_%iGHz_DX11d_POL.fits'%freq

    #freq to radio astro band name conversion
    fbc = {23:'K',33:'Ka',41:'Q',61:'V',94:'W'}

    for freq, nda in [(23, 1), (33, 1), (41, 2), (61, 2), (94, 4)]:
        #fullname has 1,2
        #wmap_bandpass_%s%i1_v5.cbp and wmap_bandpass_%s%i2_v5.cbp
        bpfname['W%03i'%freq] = 'wmap_bandpass_%s%i'%(fbc[freq],nda)

    return bpfname

