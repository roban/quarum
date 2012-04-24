import numpy
import pyfits

lambda_LyA_0 = 1215.67;
c_light_km_s = 299792.458 # km/s

lambda_CIVa_0 = 1548.1950 #;C IV  ;11.03
lambda_CIVb_0 = 1550.7700 #;C IV  ;10.73
lambda_CIV_0 = 0.5 * (lambda_CIVa_0 + lambda_CIVb_0)

def limit_mask(x, limits):
    """Return a mask selecting elements within a range.
    """
    return ((x >= numpy.min(limits)) & (x <= numpy.max(limits)))

def read_spec(fits_name):
    """Read a spectrum, automatically detecting various formats.

    Returns: wavelength, flux, error, flags, coname
    """
    if((fits_name[-5:].lower() == r'.fits') or 
       (fits_name[-4:].lower() == r'.fit')):
        if (fits_name.find(r'spSpec') > -1):
            # Read an SDSS spectrum.
            # SDSS specifications:
            # The spectrum. The first row is the spectrum, the second row is the
            # continuum subtracted spectrum, the third row is the noise in the
            # spectrum (standard deviation, in the same units as the spectrum),
            # the forth row is the mask array. The spectra are binned
            # log-linear. Units are 10^(-17) erg/cm/s^2/Ang.
            data, header = pyfits.getdata(fits_name, header=True)
            flux = data[0]
            error = data[2]
            flags = numpy.int64(data[3])

            # # 0x40000000 means "Emission line detected here" which is fine.
            # # see http://www.sdss.org/dr5/dm/flatFiles/spSpec.html#specmask
            # # for SDSS bitmask flags
            flags = flags & ~0x40000000

            redshift = float(header['Z'])
            #spec_table[3][irow] = redshift
            lambda_LyA = lambda_LyA_0 * (1. + redshift)
            lambda_CIV = lambda_CIV_0 * (1. + redshift)

            ned_name = header['OBJID']
            #spec_table[1][irow] = ned_name
            coname = ned_name

            # calculate lambda for for SDSS spectra #
            C0 = float(header['COEFF0'])
            C1 = float(header['COEFF1'])
            wavelength = 10**(C0 + C1 * numpy.arange(data.shape[1]))
        else:
            ### load fits file ###
            data, header =  pyfits.getdata(fits_name, header=True)
            coname = header['CONAME']
            wavelength = data[0,:]
            flux = data[1,:]
            error = data[2,:]
            flags = data[3,:]
    elif(fits_name[-5:].lower() == r'.spec'):
        err_name = fits_name[:-5] + '.sig'
        keckdata = numpy.loadtxt(fits_name)
        errdata = numpy.loadtxt(err_name)
        coname = None
        wavelength = 10. ** keckdata[:,0]
        flux = keckdata[:,1]
        error = errdata[:,1]
        flags = numpy.zeros(flux.shape)
    else:
        fulldata = numpy.loadtxt(fits_name)
        wavelength = fulldata[:,0]
        flux = fulldata[:,1]
        error = fulldata[:,2]
        flags = numpy.zeros(wavelength.shape)
        coname = None
    return wavelength, flux, error, flags, coname

def readin(redshift, specfile):
    """Read a spectrum, automatically detecting various formats.

    Convenience function to return slightly modified output from
    read_spec.

    Returns: rest_wavelength, wavelength, flux, error, flags
    """
    
    wavelength, flux, error, flags, coname = read_spec(specfile)
    rest_wavelength = wavelength/(1. + redshift)
    return rest_wavelength, wavelength, flux, error, flags

def read_spec_dict(specfile, sort=False):
    """Read a spectrum, automatically detecting various formats.

    Returns a dictionary with information retrieved from the file.
    """
    datadict = dict()
    if((specfile[-5:].lower() == r'.fits') or 
       (specfile[-4:].lower() == r'.fit')):
        if (specfile.find(r'spSpec') > -1):
            # Read an SDSS spectrum.
            # SDSS specifications:
            # The spectrum. The first row is the spectrum, the second row is the
            # continuum subtracted spectrum, the third row is the noise in the
            # spectrum (standard deviation, in the same units as the spectrum),
            # the forth row is the mask array. The spectra are binned
            # log-linear. Units are 10^(-17) erg/cm/s^2/Ang.
            data, header = pyfits.getdata(specfile, header=True)
            datadict['flux'] = data[0]
            datadict['error'] = data[2]
            flags = numpy.int64(data[3])

            # # 0x40000000 means "Emission line detected here" which is fine.
            # # see http://www.sdss.org/dr5/dm/flatFiles/spSpec.html#specmask
            # # for SDSS bitmask flags
            flags = flags & ~0x40000000

            datadict['flags'] = flags

            redshift = float(header['Z'])
            datadict['redshift'] = redshift
            datadict['name'] = header['OBJID']

            # calculate lambda for for SDSS spectra #
            C0 = float(header['COEFF0'])
            C1 = float(header['COEFF1'])
            wavelength = 10**(C0 + C1 * numpy.arange(data.shape[1]))
            datadict['wavelength'] = wavelength
            datadict['rest_wavelength'] = wavelength / (1. + redshift)
        else:
            ### load fits file ###
            data, header =  pyfits.getdata(specfile, header=True)
            datadict['name'] = header['CONAME']
            datadict['wavelength'] = data[0,:]
            datadict['flux'] = data[1,:]
            datadict['error'] = data[2,:]
            datadict['flags'] = data[3,:]
    elif(specfile[-5:].lower() == r'.hdf5'):
        import hdf5_utils
        datadict = hdf5_utils.load_hdf5(specfile)
    elif(specfile[-5:].lower() == r'.spec'):
        err_name = specfile[:-5] + '.sig'
        keckdata = numpy.loadtxt(specfile)
        errdata = numpy.loadtxt(err_name)
        coname = None
        datadict['wavelength'] = 10. ** keckdata[:,0]
        datadict['flux'] = keckdata[:,1]
        datadict['error'] = errdata[:,1]
        #datadict['flags'] = numpy.zeros(flux.shape)
    else:
        fulldata = numpy.loadtxt(specfile)
        datadict['wavelength'] = fulldata[:,0]
        datadict['flux'] = fulldata[:,1]
        datadict['error'] = fulldata[:,2]
        #datadict['flags'] = numpy.zeros(wavelength.shape)

    if sort:
        sorti = numpy.argsort(datadict['wavelength'])
        newdatadict = dict()
        for k, v in datadict.iteritems():
            newdatadict[k] = v
            if not numpy.isscalar(v):
                if len(v) == len(sorti):
                    newdatadict[k] = v[sorti]
        datadict = newdatadict
    return datadict

def plot_whole_spec(wavelength, flux, redshift, searchradius, plotrange,
                    axes=None):
    lambda_LyA = lambda_LyA_0 * (1. + redshift)
    lambda_CIV = lambda_CIV_0 * (1. + redshift)
    plotmask = numpy.logical_and(plotrange[0] < wavelength,
                                 plotrange[1] > wavelength)
    if axes is None:
        pylab.figure()
        axes = pylab.gca()
        pylab.plot(wavelength[plotmask], 
                   flux[plotmask] + 3. * error[plotmask], 
                   '0.65')
        pylab.plot(wavelength[plotmask], 
                   flux[plotmask] - 3. * error[plotmask], 
                   '0.65')
        pylab.plot(wavelength[plotmask], flux[plotmask], 'k')
        pylab.axvline(x=lambda_LyA, color='b', ls=':')
        pylab.axvline(x=lambda_CIV, color='r', ls=':')
        pylab.title(coname + ": z = %.2f, s/n = %.1f" % (redshift, sn_ratio))  
        pylab.savefig(graph_dir + 
                      outname + '_' + coname + '_spectrum.png')
