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
import astropy.table


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', type=str, default='',
        help = 'Optional prefix to use for output filenames.')
    parser.add_argument('--no-noise', action='store_true',
        help = 'Do not add random noise to each spectrum.')
    args = parser.parse_args()

    # We require that the SPECSIM_MODEL environment variable is set.
    if 'SPECSIM_MODEL' not in os.environ:
        raise RuntimeError('The environment variable SPECSIM_MODEL must be set.')

    # Read the template spectrum.
    template_z = 2.4
    template_file = 'spec-qso-z2.4-rmag22.62.dat'
    template_data = astropy.table.Table.read(template_file, format='ascii')
    wlen, flux = template_data['WAVELENGTH'], template_data['FLUX']
    # Extrapolate down to 2500A to allow z=3.5
    wlen_lo = np.linspace(2500., wlen[0] - 0.1, 10)
    flux_lo = np.empty_like(wlen_lo)
    flux_lo[:] = np.mean(flux[:10])
    # Extrapolate up to 34,000A to allow z=0
    wlen_hi = np.linspace(wlen[-1] + 0.1, 34000., 10)
    flux_hi = np.empty_like(wlen_hi)
    flux_hi[:] = np.mean(flux[-10:])
    # Combine the pieces.
    wlen = np.hstack([wlen_lo, wlen, wlen_hi])
    flux = np.hstack([flux_lo, flux, flux_hi])
    template = specsim.spectrum.SpectralFluxDensity(wlen, flux)

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

    # Initialize the simulation grid.
    g_grid = np.linspace(22.0, 23.0, 5)
    z_grid = np.linspace(0.5, 3.5, 31)

    # Initialize down sampling of the 0.1A simulation grid to 0.5A
    downsampling = 5
    ndown = qsim.wavelengthGrid.size // downsampling

    # Allocate output arrays.
    num_spec = len(g_grid) * len(z_grid)
    flux = np.zeros((num_cameras, num_spec, ndown))
    ivar = np.zeros_like(flux)
    wave = np.empty_like(flux)

    # Loop over g-band magnitudes and redshifts.
    spec_index = 0
    for g in g_grid:
        print('Simulating g = {:.2f}'.format(g))
        for z in z_grid:
            input_spectrum = (template
                .createRedshifted(newZ=z, oldZ=template_z)
                .createRescaled(sdssBand='g', abMagnitude=g))
            results = qsim.simulate(
                sourceType='qso', sourceSpectrum=input_spectrum,
                airmass=1.0, expTime=900., downsampling=downsampling)
            # Loop over cameras
            for camera in range(num_cameras):
                snr = (results.snr)[: ,camera]
                mask = (results.obsflux > 0) & (snr > 0)
                flux[camera, spec_index, mask] = results[mask].obsflux
                ivar[camera, spec_index, mask] = (snr[mask] / results[mask].obsflux)**2
                if not args.no_noise:
                    flux[camera, spec_index, mask] += np.random.normal(
                        scale=ivar[camera, spec_index, mask]**-0.5)
                wave[camera, spec_index] = results.wave
            spec_index += 1

    for camera, band in enumerate(bands):
        output = fitsio.FITS(args.prefix + band + '.fits', 'rw', clobber=True)
        output.write(flux[camera])
        output.write(ivar[camera])
        output.write(wave[camera])
        output.close()


if __name__ == '__main__':
    main()
