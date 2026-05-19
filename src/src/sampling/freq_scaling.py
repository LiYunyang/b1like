import numpy as np
from util import read_bandpass as rb


def freq_scaling(bandpass, beta, temp, nu0):
    """ freq_scaling.m
    [Input]
       bandpass   Experiment bandpass. Should be an array with shape [N,2],
                  where bandpass[0] is a vector of frequencies in GHz, and 
                  bandpass[1] are the corresponding spectral responses.
                  Also accepts a cell array containing bandpasses for many
                  experiments.
       beta       Power law exponent beta.
       temp       Thermodynamic temperature (in Kelvin) for a greybody model. 
                  If not specified, then a power law will be used instead.
       nu0        Optional argument. Frequency (in GHz) at which the foreground 
                  signal power is defined. The scale factor returned by this 
                  function converts a signal amplitude from that frequency to 
                  the specified bandpass.
    [Output]
      scale_fac  The appropriate scale factor to convert a foreground signal
                 amplitude from uK_CMB at frequency nu0 to uK_CMB for the
                 instrument bandpass provided.

    """
    use_greybody = False
    if temp is not None:
        use_greybody = True
    
    # Fundamental constants.
    h = 6.62606957e-34; # J*s
    kB = 1.3806488e-23; # J/K
    Tcmb = 2.72548; # K

    if bandpass.ndim > 1:
        dnu = rb.get_dnu(bandpass[0])
    
    #powerlaw
    pl = lambda v,beta: v**(2+beta)
    #greybody
    gb = lambda v,beta,T: v**(3+beta) / (np.exp(h*v*1e9 / (kB*T)) -1.0)
    #conversion for thermodynamic temp
    cf = lambda v: v**4 * np.exp( h*v*1e9 / (kB*Tcmb)) / (
                          np.exp( h*v*1e9 / (kB*Tcmb)) - 1.0)**2

    if use_greybody:
        scale_fac = gb(bandpass[0], beta, temp)
        if bandpass.ndim > 1:
          scale_fac = sum(dnu *scale_fac * bandpass[1])/(
                      sum(dnu * bandpass[1]))
        scale_fac /= gb(nu0, beta, temp)
    else: #powerlaw
        scale_fac = pl(bandpass[0], beta)
        if bandpass.ndim > 1:
          scale_fac = sum(dnu *scale_fac * bandpass[1])/(
                      sum(dnu * bandpass[1]))
        scale_fac /= pl(nu0, beta)
  
    #Calculate thermodynamic temperature conversion for target bandpass.
    conv_fac = cf(bandpass[0])
    if bandpass.ndim > 1:
        conv_fac = sum(dnu * conv_fac * bandpass[1]) /(
                sum(dnu * bandpass[1]));
    conv_fac /= cf(nu0)

    scale_fac /= conv_fac

    return scale_fac

