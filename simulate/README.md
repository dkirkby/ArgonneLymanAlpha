# Simulation Scripts

## Simulate Quasar Observations

You will need to install the `specsim` package in order to run this script:
```
pip install specsim
```
You will also need to checkout the `desimodel` from svn and set the `SPECSIM_DATA`
environment variable to point to it:
```
svn co https://desi.lbl.gov/svn/code/desimodel/trunk desimodel
cd desimodel
export SPECSIM_DATA=`pwd`
```
Then add soft links so that the `specsim` library can find the DESI throughput curves, etc:
```
cd specsim
ln -s $SPECSIM_MODEL/data/throughput specsim/data/throughput
ln -s $SPECSIM_MODEL/data/spectra specsim/data/spectra
```
Finally, copy Isabelle's
[QSO templates](https://github.com/dkirkby/ArgonneLymanAlpha/issues/1)
into this directory.

Use the following command from this directory to simulate observations on
a grid of redshifts and g-band magnitudes:
```
python qso_grid.py
```

Fit of the output of  qso_grid.py to get redshifts
```
python zfit.py  --b b.fits --r r.fits --z z.fits --t spec-qso-z2.4-rmag22.62.dat --outfile results.fits
```

Output is a binary table in second HDU with keys :
'BEST_Z' best fit redshift 
'BEST_Z_ERR' best fit redshift uncertainty
'BEST_CHI2' best fit chi2
'BEST_CHI2PDF' best fit chi2 per degree of freedom
'DELTA_CHI2' difference of chi2 between best and second best fit for a given minimal redshift separation
'BEST_SNR' total signal to noise, this is the amplitude of the template divided by its uncertainty
'TRUE_Z' true redshift of the simulated QSO spectrum
'GMAG' g-band magnitude of the input QSO spectrum
