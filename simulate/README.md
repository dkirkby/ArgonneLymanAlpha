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
