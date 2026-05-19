import numpy as np
#from astropy.io import fits
#import astropy.io as aio

# need this for bpcm class for write_outputs.py 
freqbeam = {'CMB': ('CMB',2.5), 'Dust':('Dust', 2.5), 'LT': ('LT',0)}

# dummy -- for passing into BPCM class
maporder = { 0:('CMB','sn'),
             1:('Dust', 'sn'), 
             2:('LT','sn')}

# bandpass dummy (might need real bandpass
bwidth={'CMB':20.4, 'Dust': 20.4, 'LT':1}

#bandpasses dummy
freq1 = {'CMB': 95.0, 'Dust':353.0, 'LT': 95.0}
bandpass={}
for comp in bwidth.keys():
    bw = bwidth[comp]
    frange = np.linspace(freq1[comp]-bw/2, freq1[comp]+bw/2, 100)
    bandpass[comp] = np.array([frange, frange**(-2)])

