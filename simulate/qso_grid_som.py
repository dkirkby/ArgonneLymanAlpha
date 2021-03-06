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
import write_brick


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', type=str, default='',
        help = 'Optional prefix to use for output filenames.')
    parser.add_argument('--template-file', type=str,
        default='./intemplates/QSOtemplate.fits',
        help = 'Name of file containing random QSO templates to use.')
    parser.add_argument('--template', type=int, default=None,
        help = 'Index of template to use (or randomize if not set).')
    parser.add_argument('--no-noise', action='store_true',
        help = 'Do not add random noise to each spectrum.')
    parser.add_argument('--write-bricks', action='store_true',
        help = 'Write bricks consistent with datamodel instead of simplied output.')
    parser.add_argument('-seed', type=int, default=123,
        help = 'Random number generator seed to use.')
    parser.add_argument('--zmock', type=float,
        help = 'The z each mock should be redshifted to.')
#    parser.add_argument('--g_mag', type=double,
#        help = 'The g mag each mock should be scaled to.')
    args = parser.parse_args()

    # We require that the SPECSIM_MODEL environment variable is set.
    if 'SPECSIM_MODEL' not in os.environ:
        raise RuntimeError('The environment variable SPECSIM_MODEL must be set.')

    # Set the random seed.
    generator = np.random.RandomState(args.seed)

    # Initialize the simulation grid.
    g_grid = np.linspace(22.5, 23.0, 2)
    z = args.zmock

    if args.template is not None and args.template < 0:
        # Read the z=2.4, r=22.62 template from desimodel/data/spectra/
        args.template_file = './intemplates/spec-qso-z2.4-rmag22.62.dat'
        template_data = astropy.table.Table.read(args.template_file, format='ascii')
        wlen = template_data['WAVELENGTH']
        fluxes = template_data['FLUX'][np.newaxis, :]
        template_z = 2.4
        print('Using desimodel z={:.1f} QSO template.'.format(template_z))
    else:
        # Read the template file.
#        templates = fitsio.FITS(args.template_file, mode='r')
        template_data = astropy.table.Table.read(args.template_file, format='ascii')
        wlen = template_data['WAVELENGTH']
        fluxes = template_data['FLUX'][np.newaxis, :]
#        data = templates[0].read().view('>f8').reshape((5398,2))
#        templates.close()
#        wlen = data[:, 0]
#        fluxes = data[:, 1].transpose()
        if args.template is not None:
            fluxes = fluxes[args.template:args.template+1]
        template_z = 0.0

    num_templates = len(fluxes)
    print('Using {} z={:.1f} template(s) from {}.'
        .format(num_templates, template_z, args.template_file))

    # Extend each template so it can be redshifted over the range 0.5 - 3.5.
    wlen_min = wlen[0] * (1 + template_z) / (1 + z)
    wlen_max = max(10000, wlen[-1] * (1 + template_z) / (1 + z))
    # Extrapolate down to wlen_min
    flux_lo, flux_hi = None, None
    if wlen_min < wlen[0]:
        print('Extrapolate down to z={:.1f} at {:0f}A.'.format(z, wlen_min))
        wlen_lo = np.linspace(wlen_min - 1, wlen[0] - 0.1, 10)
        flux_lo = np.empty_like(wlen_lo)
        wlen = np.hstack([wlen_lo, wlen])
    # Extrapolate up to wlen_max
    if wlen_max > wlen[-1]:
        print('Extrapolate up to z={:.1f} at {:0f}A.'.format(z, wlen_max))
        wlen_hi = np.linspace(wlen[-1] + 0.1, wlen_max + 1, 10)
        flux_hi = np.empty_like(wlen_hi)
        wlen = np.hstack([wlen, wlen_hi])
    # Prepare each template.
    templates = [ ]
    for flux in fluxes:
        # Use the average flux at each end of the spectrum for extraploation.
        # Combine the pieces.
        if flux_lo is not None:
            flux_lo[:] = np.mean(flux[:10])
            flux = np.hstack([flux_lo, flux])
        if flux_hi is not None:
            flux_hi[:] = np.mean(flux[-10:])
            flux = np.hstack([flux, flux_hi])
        template = specsim.spectrum.SpectralFluxDensity(wlen, flux)
        templates.append(template)

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

    # Initialize down sampling of the 0.1A simulation grid to 0.5A
    downsampling = 10
    ndown = qsim.wavelengthGrid.size // downsampling

    # Allocate output arrays.
    num_spec = len(g_grid)
    flux = np.zeros((num_cameras, num_spec, ndown))
    ivar = np.zeros_like(flux)
    wave = np.empty_like(flux)
    true_z = np.empty((num_cameras,num_spec))
    g_band_mag = np.empty_like(true_z)
    r_band_mag = np.empty_like(true_z)
    z_band_mag = np.empty_like(true_z)
    W1_band_mag = np.empty_like(true_z)
    W2_band_mag = np.empty_like(true_z)
    
    # Loop over g-band magnitudes and redshifts.
    spec_index = 0
    for g in g_grid:
        print('Simulating g = {:.2f}'.format(g))
        #        for z in z_grid:
        # Pick a random template to use. We do not use np.random.choice()
        # for Julien's benefit.
        template_index = int(generator.uniform()*num_templates)
        template = templates[template_index]
        # Run the simulation.
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

            true_z[camera, spec_index] = z
            g_band_mag[camera, spec_index] = g
            r_band_mag[camera, spec_index] = g
            z_band_mag[camera, spec_index] = g
            W1_band_mag[camera, spec_index] = g
            W2_band_mag[camera, spec_index] = g
        spec_index += 1

    for camera, band in enumerate(bands):
        if args.write_bricks:
            write_brick.write_brick_file(
                band=band,brickname='1234p567',NSpectra=num_spec,NWavelength=ndown,
                Flux=flux[camera],InvVar=ivar[camera],Wavelength=wave[camera],
                Resolution=wave[camera],TrueZ=true_z[camera],GBand=g_band_mag[camera],
                RBand=r_band_mag[camera],ZBand=z_band_mag[camera],
                W1Band=W1_band_mag[camera],W2Band=W2_band_mag[camera])
        else:
            output = fitsio.FITS(args.prefix + band + '.fits', 'rw', clobber=True)
            output.write(flux[camera])
            output.write(ivar[camera])
            output.write(wave[camera])
            output.close()


if __name__ == '__main__':
    main()
