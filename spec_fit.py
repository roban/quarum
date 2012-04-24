import argparse

import numpy
numpy.seterr(invalid='ignore', divide='ignore')

from quarum import powergauss
from quarum import spec_file_utils

import cPickle as pickle

lambda_LyA_0 = 1215.67
lambda_CIVa_0 = 1548.1950
lambda_CIVb_0 = 1550.7700
lambda_CIV_0 = 0.5 * (lambda_CIVa_0 + lambda_CIVb_0)
lambda_NV_0 = 1240.81

def snint(string):
    """Convert scientific notation string to int.
    """
    value = int(float(string))
    return value

def readin(redshift, specfile):
    """Read in a spectrum file."""
    data = spec_file_utils.read_spec_dict(specfile)
    data['rest_wavelength'] = data['wavelength'] / (1. + redshift)
    if 'flags' not in data:
        data['flags'] = numpy.zeros(data['wavelength'].shape)
    return (data['rest_wavelength'], data['wavelength'], data['flux'], 
            data['error'], data['flags'])

def spec_fit(rest_wavelength, flux, error, goodmask, nsamples,
             prethin, burn, filename, 
             n_components, add_dampingwing, convolve_width_pix, single_NV, 
             input_model, input_model_trim, 
             vlim = 1.5e4, 
             forest_grid=True, 
             forest_v_min=-3000.0, forest_v_min_Lya=None, forest_v_max=1005,
             forest_v_spacing = 500., # spacing between absorbers (km/s)
             forest_sigma_spacing = 5., # prior sigma for that spacing
             forest_metal_spacing = 2, # Add metal lines to some absorbers.
             cov_prior_args=None, alpha=None):
    """Fit a model to a spectrum."""

    linelist = numpy.loadtxt('data/line_list.dat', usecols=(0,))

    if single_NV:
        line_centers=(powergauss.lambda_LyA_0, powergauss.lambda_CIV_0)
        linelist = linelist.tolist() + [powergauss.lambda_NV_0,]
    else:
        line_centers=(powergauss.lambda_LyA_0, powergauss.lambda_CIV_0, 
                      powergauss.lambda_NV_0)
    linelist = numpy.sort(linelist).tolist()

    # # Remove lines more than vlim km/s outside of the observed
    # # wavelength range.
    # lambda_min = (numpy.min(rest_wavelength) / 
    #               (1 + vlim * 1e5/cc.c_light_cm_s))
    # lambda_max = (numpy.min(rest_wavelength) / 
    #               (1 - vlim * 1e5/cc.c_light_cm_s))
    # linelist = filter(lambda l: (l > lambda_min) and (l < lambda_max), linelist)

    startvalues = {}
    ranges = {}

    # Fix powerlaw index of damping wing if specified.
    if alpha is not None:
        startvalues['dw_alpha'] = alpha
        ranges['dw_alpha'] = 'fixed'

    # Change priors for NV line
    startvalues['ewidth_4'] = 3000.
    ranges['ewidth_4'] = (500.0, 8000.)
    ranges['evshift_4'] = (-3000., 500.)

    if input_model is not None:
        imodel = powergauss.PowerGaussMCMC.load(input_model)
        startvalues.update(imodel.min_deviance_params(trim=input_model_trim))
        ranges.update(dict(map(lambda k: (k, 'fixed'), startvalues.keys())))


        # Now compensate for the different flux normalization factor
        # that will be applied during the fit.
        relscale = numpy.median(flux)/imodel.scale
        for (k,v) in startvalues.iteritems():
            if (k.startswith('peak') 
                or k.startswith('epeak') 
                or k == 'contflux'):
                startvalues[k] = v / relscale

    # Set up starting values and ranges for absorption line shifts.
    absorber_centers = []
    if forest_grid:
        if forest_v_min_Lya is None:
            forest_v_min_Lya = powergauss.wavelength_to_vshift(
                numpy.min(rest_wavelength), [powergauss.lambda_LyA_0,])[0]
        v_min = min(forest_v_min_Lya, forest_v_min)
        forest_v_spacing = float(forest_v_spacing)
        forest_sigma_shift = forest_v_spacing / forest_sigma_spacing
        n_forest = int((forest_v_max-v_min) / forest_v_spacing)
        for j in xrange(0, n_forest):
            # Velocity of absorption system:
            v_center = (forest_v_max - j * forest_v_spacing)
            # Add Ly-alpha component, if appropriate.
            if v_center >= forest_v_min_Lya:
                absorber_centers.append([powergauss.lambda_LyA_0])
            else:
                absorber_centers.append([])
            # Add metal components, if appropriate.
            if j % forest_metal_spacing == 0:
                if v_center >= forest_v_min:
                    absorber_centers[-1].extend([powergauss.lamdba_NVa_0, 
                                                 powergauss.lambda_NVb_0,
                                                 powergauss.lambda_CIVa_0,
                                                 powergauss.lambda_CIVb_0])
            for i in xrange(len(absorber_centers[j])):
                startvalues['avshift_%i_%i'%(j,i)] = v_center
                ranges['avshift_%i_%i'%(j,i)] = (
                    'normal', v_center, forest_sigma_shift)
                if j % forest_metal_spacing == 0:
                    # Let the metalic absorbers be wider...
                    ranges['awidth_%i_%i'%(j,i)] = [3., 200.]
                    # ...and shift more.
                    ranges['avshift_%i_%i'%(j,i)] = (
                        'normal', 
                        startvalues['avshift_%i_%i'%(j,i)], 
                        forest_metal_spacing * forest_sigma_shift)


    """Define a continuum plus line(s) model."""
    fmodel = powergauss.PowerGaussMCMC(
        filename=filename,
        wavelength=rest_wavelength,
        error=error, 
        observed_flux=flux, 
        goodmask=goodmask,
        observed=True, 
        extra_line_centers=linelist,
        line_centers=line_centers,
        n_components=n_components,
        add_dampingwing=add_dampingwing,
        convolve_width_pix=convolve_width_pix,
        startvalues = startvalues,
        ranges = ranges,
        n_absorbers = n_forest,
        absorber_centers = absorber_centers,
        cov_prior_args=cov_prior_args,
        )

    """Fit the model to the spectrum."""
    fmodel.sample(nsamples, verbose=True, thin=prethin, burn=burn)

    return fmodel

if __name__ == '__main__':
    ### Argument parsing. ###
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", "--nsamples", action='store', 
                        dest="nsamples", default=2e5, type=snint,
                        help="Number of samples to run.")

    parser.add_argument("-b", "--burn", action='store', 
                        dest="burn", default=0, type=snint,
                        help="Number of samples to burn at start.")

    parser.add_argument("-p", "--prethin", action='store', 
                        dest="prethin", default=10, type=snint,
                        help="Factor by which to thin samples during fit.")

    parser.add_argument("-s", "--specfile", action='store', 
                        dest="specfile",
                        help="File containing spectrum.")

    parser.add_argument("-o", "--outfile", action='store', 
                        dest="outfile",
                        help="File for output of fit results.")

    parser.add_argument("-z", "--redshift", action='store', 
                        default=0.0,
                        dest="redshift", type=float,
                        help="Redshift of the target in the spectrum.")

    parser.add_argument("-c", "--component_number", action='store', 
                        dest="n_components", default=2, type=snint,
                        help="Number of components for primary lines.")

    parser.add_argument("--single_NV", action='store_true', 
                        dest="single_NV", default=False, 
                        help="Use only a single component for NV 1240 Ang.")

    parser.add_argument("-d", "--dampingwing", action='store_true', 
                        dest="add_dampingwing", default=False, 
                        help="Add a damping wing to the fit.")

    parser.add_argument("-a", "--alpha", action='store', 
                        dest="alpha", default=None, type=float,
                        help="Index of damping wing powerlaw (fixed to specified value, otherwise free).")

    parser.add_argument("-l", "--lower_lambda", action='store', 
                        dest="lower_lambda", default=1000., type=float,
                        help="Minimum of wavelength range to use.")

    parser.add_argument("-u", "--upper_lambda", action='store', 
                        dest="upper_lambda", default=2000., type=float,
                        help="Maximum of wavelength range to use.")

    parser.add_argument("-k", "--convolve_width_pix", action='store', 
                        dest="convolve_width_pix", default=None, type=float,
                        help="Width in pixels of convolution kernel.")

    parser.add_argument("-i", "--input_model", action='store', 
                        dest="input_model", default=None, type=str,
                        help="File name of a model to use to fix parameters.")

    parser.add_argument("--input_model_trim", action='store', 
                        dest="input_model_trim", default=1000, type=int,
                        help="Number of iterations to trim from input model.")

    parser.add_argument("--noforest", action='store_true', default=False, 
                        help="No Lyman-alpha forest lines.")

    parser.add_argument("--cov_prior_args", action='store', 
                        dest="cov_file", default=None,
                        help="File containing MvNormal priors.")


    options = parser.parse_args()

    if options.cov_file is not None:
        input = open(options.cov_file, 'r')
        cov_prior_args = pickle.load(input)
        input.close()
    else:
        cov_prior_args = None


    rest_wavelength, wavelength, flux, error, flags = readin(
        specfile=options.specfile,
        redshift=options.redshift)
    mask = ((rest_wavelength <= options.upper_lambda) &
            (rest_wavelength >= options.lower_lambda))
    if not numpy.all(mask):
        assert numpy.sum(mask) > 0
        print "Masking out wavelengths outside %.1f to %.1f Angstroms:" % (
            options.lower_lambda, options.upper_lambda)
        print " fraction of original spectrum included = %f" % (
            float(numpy.sum(mask))/len(mask))
    rest_wavelength = rest_wavelength[mask]
    flux = flux[mask]
    error = error[mask]
    goodmask = (flags[mask] == 0)
    goodmask = (goodmask & (error>0))
    fmodel = spec_fit(rest_wavelength=rest_wavelength, 
                      flux=flux, 
                      error=error,
                      goodmask=goodmask,
                      nsamples=options.nsamples,
                      prethin=options.prethin,
                      burn=options.burn,
                      filename=options.outfile + '.mcmc',
                      n_components=options.n_components,
                      add_dampingwing=options.add_dampingwing,
                      single_NV=options.single_NV,
                      convolve_width_pix = options.convolve_width_pix,
                      input_model = options.input_model,
                      input_model_trim = options.input_model_trim,
                      forest_grid = not(options.noforest),
                      cov_prior_args=cov_prior_args,
                      alpha=options.alpha,
                      )
    
