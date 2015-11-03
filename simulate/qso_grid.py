#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a MIT style license - see LICENSE.rst
"""
Simulate QSO spectra on a redshift, g-magnitude grid.
"""
from __future__ import print_function, division

import os
import argparse
import specsim
import numpy as np
import fitsio


def main():
    # We require that the SPECSIM_MODEL environment variable is set.
    if 'SPECSIM_MODEL' not in os.environ:
        raise RuntimeError('The environment variable SPECSIM_MODEL must be set.')

    # Read the template spectrum.
    template_z = 2.4
    template_file = 'spec-qso-z2.4-rmag22.62.dat'
    template = specsim.spectrum.SpectralFluxDensity.loadFromTextFile(template_file,
        wavelengthColumn=0,valuesColumn=1, extrapolatedValue=0.)

    # Create the default atmosphere for the requested sky conditions.
    atmosphere = specsim.atmosphere.Atmosphere(
        skyConditions='dark', basePath=os.environ['SPECSIM_MODEL'])

    # Create a quick simulator using the default instrument model.
    qsim = specsim.quick.Quick(
        atmosphere=atmosphere, basePath=os.environ['SPECSIM_MODEL'])

    # Specify the simulation wavelength grid to use (in Angstroms).
    qsim.setWavelengthGrid(3500.3, 9999.7, 0.1)

    bands = 'brz'
    num_cameras = len(bands)
    flux = None

    downsampling = 5
    ndown = qsim.wavelengthGrid.size // downsampling
    last = ndown*downsampling

    # Loop over g-band magnitudes and redshifts.
    g_grid = np.linspace(22.0, 23.0, 5)
    z_grid = np.linspace(1.0, 3.5, 11)
    spec_index = 0
    for g in g_grid:
        print('Simulating g = {:.2f}'.format(g))
        for z in z_grid:
            input_spectrum = (template
                .createRescaled(sdssBand='g', abMagnitude=g)
                .createRedshifted(newZ=z, oldZ=template_z))
            results = qsim.simulate(
                sourceType='qso', sourceSpectrum=input_spectrum,
                airmass=1.2,expTime=900.,downsampling=downsampling)
            # Allocate output arrays if necessary.
            if flux is None:
                num_spec = len(g_grid) * len(z_grid)
                num_wlen = len(results.wave)
                flux = np.zeros((num_cameras, num_spec, num_wlen))
                ivar = np.zeros_like(flux)
                wave = np.empty_like(flux)
                wave[:,:] = results.wave
            # Loop over cameras
            for camera in range(num_cameras):
                nphotons = (results.nobj)[:,camera]
                nphotons_var = (nphotons + (results.nsky)[:,camera] +
                    (results.rdnoise)[:,camera]**2 + (results.dknoise)[:,camera]**2)
                mask = nphotons_var > 0
                # Add Poisson noise.
                nphotons[mask] += np.random.normal(scale=np.sqrt(nphotons_var[mask]))
                # Apply flux calibration.
                throughput = qsim.cameras[camera].sourceCalib[:last:downsampling]
                mask = mask & (throughput > 0)
                flux[camera, spec_index, mask] = nphotons[mask] / throughput[mask]
                ivar[camera, spec_index, mask] = nphotons_var[mask] / throughput[mask]

    for camera, band in enumerate(bands):
        output = fitsio.FITS(band + '.fits', 'rw', clobber=True)
        output.write(flux[camera])
        output.write(ivar[camera])
        output.write(wave[camera])
        output.close()


if __name__ == '__main__':
    main()
