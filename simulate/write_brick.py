from astropy.io import fits

def write_brick_file(band, brickname,
                     Flux, InvVar, Wavelength, Resolution, truth):
    """
    Write brick file

    HDU0 	FLUX 	IMAGE 	Spectral flux [nspec, nwave]
    HDU1 	IVAR 	IMAGE 	Inverse variance of flux
    HDU2 	WAVELENGTH 	IMAGE 	1D common wavelength grid in Angstroms
    HDU3 	RESOLUTION 	IMAGE 	3D sparse resolution matrix data [nspec,ndiag,nwave]
    HDU4 	FIBERMAP 	BINTABLE 	Fibermap entries

    HDU0
    NAXIS1 	3494 	int 	Number of wavelength bins
    NAXIS2 	51 	int 	Number of spectra
    EXTNAME 	FLUX 	str 	erg/s/cm^2/Angstrom

    HDU1
    NAXIS1 	3494 	int 	Number of wavelength bins
    NAXIS2 	51 	int 	Number of spectra
    EXTNAME 	IVAR 	str 	1 / (erg/s/cm^2/A)^2

    HDU2
    NAXIS1 	3494 	int 	Number of wavelength bins
    NAXIS2 	51 	int 	Number of spectra
    EXTNAME 	IVAR 	str 	1 / (erg/s/cm^2/A)^2

    HDU3
    NAXIS1 	3494 	int 	Number of wavelength bins
    NAXIS2 	21 	int 	Number of diagonals
    NAXIS3 	51 	int 	Number of spectra
    EXTNAME 	RESOLUTION 	str 	no dimension

    HDU4
    NAXIS1 	224 	int 	length of dimension 1
    NAXIS2 	51 	int 	Number of spectra
    """

    NSpectra, NWavelength = Flux.shape

    outfile = 'brick-%s-%s.fits' % (band,brickname)

    head0 = fits.Header()
    head0.append(card=('NAXIS1',NWavelength,'Number of wavelength bins'))
    head0.append(card=('NAXIS2',NSpectra,'Number of spectra'))
    head0.append(card=('EXTNAME','FLUX','erg/s/cm^2/Angstrom'))

    hdu0 = fits.PrimaryHDU(data=Flux,header=head0)

    hdu1 = fits.ImageHDU(data=InvVar,name='IVAR')

    hdu2 = fits.ImageHDU(data=Wavelength,name='WAVELENGTH')

    hdu3 = fits.ImageHDU(data=Wavelength,name='RESOLUTION')

# Create HDU4

    hdu4 = fits.TableHDU(name='FIBERMAP')

    col1 = fits.Column(name='TRUEZ',format='D',array=truth['TRUEZ'])
    col2 = fits.Column(name='GBANDT',format='D',array=truth['GBANDT'])
    col3 = fits.Column(name='RBANDT',format='D',array=truth['RBANDT'])
    col4 = fits.Column(name='ZBANDT',format='D',array=truth['ZBANDT'])
    col5 = fits.Column(name='W1BANDT',format='D',array=truth['W1BANDT'])
    col6 = fits.Column(name='W2BANDT',format='D',array=truth['W2BANDT'])
    col7 = fits.Column(name='TMPID',format='I',array=truth['TMPID'])

    hdu4 = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7])

    hdulist = fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
    hdulist.writeto(outfile,clobber=True)
