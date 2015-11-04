#!/usr/bin/env python   
#########################################
#
# Welcome to Variable Quasar Simulator 
#
#########################################

import numpy as np
import scipy as sp
import random
import astropy.io.fits as fits
from astropy.io import ascii
from matplotlib import pyplot as p
from argparse import ArgumentParser
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import arange, array, exp

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

### ARGUMENTS ###
parser = ArgumentParser()
parser.add_argument("-i", "--in_spec", help="Input spectrum",type=str,default=None,metavar="INSPEC")
parser.add_argument("-o", "--out_spec", help="Output spectrum",type=str,default='var_qso.dat',metavar="OUTSPEC")
parser.add_argument("-t", "--time", help="Delta t between input and output in years",type=float,default=1.,metavar="TIME")
parser.add_argument("-v", "--var_selec", help="Quasar was selected by variability",type=bool,default=False,metavar="VARSELEC")
parser.add_argument("-p", "--plot", help="Plot result",type=bool,default=False,metavar="PLOT")
arg = parser.parse_args()
in_spec = arg.in_spec
out_spec = arg.out_spec
Dt = arg.time
var_selec = arg.var_selec
plot = arg.plot
#
# Load known QSO
#
h = fits.open('Selection_superset.fits')
known_qso = h[1].data
#
# Read Spectrum
#
spec = ascii.read(in_spec)
#
# Get Structure function parameters
#
i = random.randint(0,len(known_qso))
A = [known_qso[i]['A_u'],known_qso[i]['A_3B_g'],known_qso[i]['A_3B_r'],known_qso[i]['A_3B_i'],known_qso[i]['A_z']]
gamma = known_qso[i]['gamma_3B']
#
# Choose if positive or negative fluctuation
#
sign = random.choice([-1,1])
#
# Compute variation in flux
#
flux_fact = [0.,0.,0.,0.,0.]
for i in np.arange(5):
    dmag = sign*A[i]*Dt**gamma
    flux_fact[i] = 10**(-dmag/2.5)
#
# Interpolate
#
filter_mean = [3557,4825,6261,7672,9097]
interp = interp1d(filter_mean,flux_fact)
extrap = extrap1d(interp)
#
# Output spectra
#
wave = spec['WAVELENGTH']
vari_flux = spec['FLUX']*extrap(wave)
#
# Plot
#
if(plot):
    p.plot(wave,spec['FLUX'],label='input')
    p.plot(wave,vari_flux,label='vari')
    p.plot(wave,extrap(wave),label='ratio')
    p.legend()
    p.show()
#
# Write output
#
spec['FLUX'] = vari_flux
ascii.write(spec,out_spec)
