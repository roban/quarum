"""Spectral models with powerlaw continuum plus Gaussian emission lines.
"""
import os
import cPickle as pickle
import itertools
from collections import defaultdict

import numpy
import scipy
import scipy.signal
numpy.seterr(invalid='ignore', divide='ignore')
import pymc

import cosmolopy.constants as cc

lambda_NV_0 = 1240.81
lambda_LyA_0 = 1215.67
lambda_SiII_0 = 1262.59

# from Verner et al. 1994 A&AS 108 287 Table1
lamdba_NVa_0 = 1238.8210 #;N V   ;10.34
lambda_NVb_0 = 1242.8040 #;N V   ;10.04
lambda_CIVa_0 = 1548.1950 #;C IV  ;11.03
lambda_CIVb_0 = 1550.7700 #;C IV  ;10.73
lambda_CIV_0 = 0.5 * (lambda_CIVa_0 + lambda_CIVb_0)

class PowerGauss(object):
    """Probabilistic spectral model with powerlaw continuum plus
    Gaussian-profile emission lines.

    The error in the measured flux is assumed to be Gaussian.

    Some additional emission and absorption components can optionally
    be included.
    
    """
    def __init__(self, wavelength, error, observed_flux,
                 goodmask=None, observed=True,
                 startvalues = {}, ranges = {},
                 n_components=2, 
                 line_centers=(lambda_LyA_0, lambda_CIV_0, lambda_NV_0),
                 extra_line_centers=(),
                 add_dampingwing=False,
                 absorber_centers=(),
                 n_absorbers=0,
                 convolve_width_pix = None,
                 scale=None,
                 cov_prior_args = None,
                 pivot_wavelength = lambda_LyA_0,
                 ):
        numpy.seterr(invalid='ignore', divide='ignore')

        ### Record arguments and set defaults. ###
        # The observed data.
        self.wavelength = wavelength
        self.error = error
        self.observed_flux = observed_flux
        self.goodmask = goodmask

        # Make a default mask (all ones), if needed.
        if self.goodmask is None:
            self.goodmask = numpy.ones_like(self.observed_flux, 
                                            dtype=numpy.bool)

        # Scale the flux for numerical stability.
        if scale is None:
            # Default flux scale (normalization) is the median flux.
            self.scale = numpy.median(observed_flux)
        else:
            self.scale = scale
        self.scaled_flux = self.observed_flux/self.scale
        self.scaled_error = self.error/self.scale

        self.n_components = n_components
        self.line_centers = line_centers
        self.extra_line_centers = extra_line_centers
        self.n_absorbers = n_absorbers
        self.absorber_centers = absorber_centers
        self.pivot_wavelength = pivot_wavelength
        self.add_dampingwing = add_dampingwing
        self.convolve_width_pix = convolve_width_pix

        ### Set up parameters and defaults for flux components. ###

        # Number of lines:
        self.n_lines = len(self.line_centers)
        # Total number of components in all lines:
        self.n_gauss = self.n_lines * self.n_components

        # Set up central wavelengths of the primary line emission
        # components. Ordered by line first, then component.
        self.centers = (numpy.ones((self.n_components, self.n_lines)) 
                        * self.line_centers
                        ).transpose().flatten()

        # Add in the extra lines.
        self.all_centers = numpy.array(list(self.centers) 
                                       + list(self.extra_line_centers))

        # Set up default starting values for primary lines.
        startvalues_defaults = dict(
            contindex = -0.1,
            contflux = numpy.median(self.scaled_flux),
            )
        median_scaled_flux = numpy.median(self.scaled_flux)
        for i in xrange(self.n_lines):
            for j in xrange(self.n_components):
                k = (i * self.n_components) + j
                startvalues_defaults['width_%i'%k] = (
                    1000.0 + (j * 2000. / self.n_components))
                startvalues_defaults['peak_%i'%k] = (
                    self.closest_flux(self.centers[k]) 
                    - numpy.median(self.scaled_flux))
                if startvalues_defaults['peak_%i'%k] < 0.1 * median_scaled_flux:
                    startvalues_defaults['peak_%i'%k] = 0.1 * median_scaled_flux
                startvalues_defaults['vshift_%i'%k] = -1000. - (300. * j)

        # Set up default parameter ranges for primary lines.
        ranges_defaults = dict(contflux = (0.0, numpy.max(self.scaled_flux)),
                               )
        for i in xrange(self.n_gauss):
            ranges_defaults['width_%i'%i] = (300., 8000.)
            ranges_defaults['peak_%i'%i] = (0.0, 
                                            3. * numpy.max(self.scaled_flux))
            ranges_defaults['vshift_%i'%i] = (-6000., 4000.)

        # Now set up default startvalues and ranges for extra lines.
        self.n_extra_lines = len(self.extra_line_centers)
        for i in xrange(self.n_extra_lines):
            #print i
            startvalues_defaults['ewidth_%i'%i] = 1000.0
            startvalues_defaults['epeak_%i'%i] = (
                self.closest_flux(self.extra_line_centers[i]) 
                - median_scaled_flux)
            if startvalues_defaults['epeak_%i'%i] < 0.1 * median_scaled_flux:
                startvalues_defaults['epeak_%i'%i] = 0.1 * median_scaled_flux
            startvalues_defaults['evshift_%i'%i] = -1000.0

        # Set up default ranges for extra lines.
        for i in xrange(self.n_extra_lines):
            ranges_defaults['ewidth_%i'%i] = (500.0, 3000.)
            ranges_defaults['epeak_%i'%i] = (0.0, 
                                             3. * numpy.max(self.scaled_flux))
            ranges_defaults['evshift_%i'%i] = (-3000., 500.)

        # Set up default starting values and ranges for damping wing
        # parameters.
        startvalues_defaults['dw_logtau0'] = -1.0
        startvalues_defaults['dw_lambda_e'] = 1210.
        startvalues_defaults['dw_alpha'] = -1.25
        ranges_defaults['dw_logtau0'] = [-4, 0.3]
        ranges_defaults['dw_lambda_e'] = [1175., lambda_LyA_0]
        ranges_defaults['dw_alpha'] = [-2., -1.]

        # Set up default starting values and ranges for absorption lines.
        for j in xrange(self.n_absorbers):
            for i in xrange(len(self.absorber_centers[j])):
                startvalues_defaults['awidth_%i_%i'%(j,i)] = 10.
                ranges_defaults['awidth_%i_%i'%(j,i)] = [1., 100.]
                startvalues_defaults['apeak_%i_%i'%(j,i)] = (
                    self.closest_flux(self.absorber_centers[j][i]) 
                    - median_scaled_flux)
                if startvalues_defaults['apeak_%i_%i'%(j,i)] > -0.1:
                    startvalues_defaults['apeak_%i_%i'%(j,i)] = -1.0
                ranges_defaults['apeak_%i_%i'%(j,i)] = [-20.0, 0.0]
                startvalues_defaults['avshift_%i_%i'%(j,i)] = -100. + j * 100.
                ranges_defaults['avshift_%i_%i'%(j,i)] = [-2000., 2000.]

                
        ### Override default start values and ranges with passed values. ###
        startvalues_defaults.update(startvalues)
        startvalues = startvalues_defaults
        self.startvalues = startvalues
        ranges_defaults.update(ranges)
        ranges = ranges_defaults
        self.ranges = ranges

        ### Define model variables (pymc Stochastics, etc.). ###

        # Continuum powerlaw index and continuum flux scale.
        if observed:
            self.contindex = make_stochastic('contindex', 
                                             startvalues,
                                             ranges)
            self.contflux = make_stochastic('contflux', 
                                            startvalues,
                                            ranges)
        else:
            self.contindex = startvalues['contindex']
            self.contflux = startvalues['contflux']

        # Set up convolution kernel and pad wavelength.
        if self.convolve_width_pix is not None:
            # Pick an odd kernel size encompasing at least 6 sigma on
            # either side of center.
            ksize = numpy.ceil(12 * convolve_width_pix)
            if ksize % 2 == 0:
                ksize += 1
            self.kernel = scipy.signal.gaussian(ksize, convolve_width_pix)
            self.kernel /= numpy.sum(self.kernel)
            kcenter = len(self.kernel) / 2
            lowpoints = (self.wavelength[0]
                         + (self.wavelength[0] - self.wavelength[1])
                         * (kcenter - numpy.arange(kcenter)))
            highpoints = (self.wavelength[-1]
                          + (self.wavelength[-1] - self.wavelength[-2])
                          * (1 + numpy.arange(kcenter)))
            self.model_wavelength = numpy.hstack(
                (lowpoints, self.wavelength, highpoints))
        else:
            self.model_wavelength = self.wavelength

        # Set up gaussian widths, peaks, velocity shifts for the
        # primary lines.
        self.widths = []
        self.peaks = []
        self.vshifts = []
        self.order_potentials = []
        for i in xrange(self.n_lines):
            for j in xrange(self.n_components):
                k = (i * self.n_components) + j
                for basename in ['width', 'peak', 'vshift']:
                    pname = basename + '_%i'%k
                    if observed:
                        param = make_stochastic(pname, 
                                                startvalues,
                                                ranges)
                    else:
                        # If there's no data, don't make a stochastic,
                        # just fix the value at the startvalue.
                        param = startvalues[pname]
                    # Add lines to self.widths, self.peaks, self.vshifts
                    self.__dict__[basename+'s'].append(param)

            # Now set up a Potential inforcing strict ordering of widths.
            # This is to reduce the degeneracy of the model.
            starti = (i * self.n_components)
            endi = starti + self.n_components
            x = self.widths[starti:endi]
            self.order_potentials.append(
                pymc.Potential(logp=check_order,
                               doc="Potential inforcing ordering of widths.",
                               name="order_potential_%i"%i,
                               parents=dict(x=x)))

        # Set up Gaussian parameters for extra lines.
        for i in xrange(self.n_extra_lines):
            for basename in ['ewidth', 'epeak', 'evshift']:
                pname = basename + '_%i'%i
                if observed:
                    param = make_stochastic(pname,
                                            startvalues,
                                            ranges)
                else:
                    param = startvalues[pname]
                # Add extra lines to self.widths, self.peaks, self.vshifts
                self.__dict__[basename[1:]+'s'].append(param)

        # Set up gaussian widths, peaks, velocity shifts for the
        # absorption lines.
        self.awidths = []
        self.apeaks = []
        self.avshifts = []
        self.all_centers = list(self.all_centers)
        # Loop over the absorption systems.
        for j in xrange(self.n_absorbers): 
            # Loop over the different lines in each system.
            for i in xrange(len(self.absorber_centers[j])):
                # Add the central wavelength to the list.
                self.all_centers.append(self.absorber_centers[j][i])
                for basename in ['awidth', 'apeak', 'avshift']:
                    pname = basename + '_%i_%i'%(j,i)
                    if i>0 and basename is 'avshift':
                        # shifts are linked, use the same shift.
                        p0name = basename + '_%i_%i'%(j,0)
                        param = self.__dict__[p0name]
                    else:
                        # If on the first component, make a new stochastic.
                        param = make_stochastic(pname, 
                                                startvalues,
                                                ranges)
                        self.__dict__[pname] = param
                    # Add lines to self.awidths, self.apeaks, self.avshifts
                    self.__dict__[basename+'s'].append(param)
        self.all_centers = numpy.array(self.all_centers)
        self.widths.extend(self.awidths)
        self.peaks.extend(self.apeaks)
        self.vshifts.extend(self.avshifts)

        # Create separate deterministics for each flux component to
        # make updating individual component parameters faster.
        self.components = []
        self.component_shapes = []
        # First the continuum.
        self.components.append(pymc.Deterministic(
                eval=self.continuum_component,
                doc=self.continuum_component.__doc__,
                name='flux_component_cont',
                parents=dict(wavelength=self.model_wavelength,
                             pivot_wavelength=self.pivot_wavelength, 
                             contindex=self.contindex, 
                             contflux=self.contflux),
                trace=False,
                cache_depth=2))
        # Now all the Gaussian line components.
        for i in xrange(len(self.all_centers)):
            # First we get just the shape, then later scale it, to
            # reduce the number of times we need to evaluate the
            # shape.
            self.component_shapes.append(
                pymc.Deterministic(
                    eval=self.line_component,
                    doc=self.line_component.__doc__,
                    name='flux_component_shape_%i' % i,
                    parents=dict(
                        wavelength=self.model_wavelength,
                        center=self.all_centers[i],
                        width=self.widths[i], 
                        peak=1.0, 
                        vshift=self.vshifts[i]),
                    trace=False,
                    cache_depth=2))
            # Then we scale the shape.
            component = self.peaks[i] * self.component_shapes[-1]
            component.__name__ = 'flux_component_%i' % i
            self.components.append(component)

        # Set up the true flux stochastic (flux without noise), with
        # or without a damping wing factor.
            
        # For speed, I've split the addition into a binary tree, which
        # dramatically reduces the number of addition operations to do
        # when only a single component is updated. All the complicated
        # setup happens at initialization. After that, pymc will
        # figure out which parts of the tree to update when each
        # parameter changes.

        # Number of tree levels.
        nlevels = int(1 + numpy.ceil(numpy.log2(len(self.components))))

        # Start with all the flux components as the leaf nodes.
        self._trueflux_levels = [[]]
        self._trueflux_levels[0].extend(self.components[:])
        for ilevel in xrange(1,nlevels):
            # Number of nodes in previous level.
            noldnodes = len(self._trueflux_levels[ilevel-1])
            # Number of nodes in this level.
            nnewnodes = int(noldnodes/2) + noldnodes%2
            # Loop through pairs.
            self._trueflux_levels.append([])
            for inode in xrange(1, noldnodes, 2):
                # Add two nodes from the previous level.
                self._trueflux_levels[ilevel].append(
                    self._trueflux_levels[ilevel-1][inode-1] +
                    self._trueflux_levels[ilevel-1][inode])
            # Pass along the end node if we have an odd number.
            if noldnodes%2 > 0:
                self._trueflux_levels[ilevel].append(
                    self._trueflux_levels[ilevel-1][-1])
            assert len(self._trueflux_levels[ilevel]) == nnewnodes

        if add_dampingwing:
            # Set up damping wing parameters.
            self.dw_lambda_0 = lambda_LyA_0
            self.dw_logtau0 = make_stochastic('dw_logtau0', 
                                              startvalues, 
                                              ranges)
            self.dw_tau0 = 10**self.dw_logtau0
            self.dw_tau0.__name__ = 'dw_tau0'
            self.dw_tau0.keep_trace = True
            self.dw_lambda_e = make_stochastic('dw_lambda_e', startvalues, 
                                               ranges)
            self.dw_alpha = make_stochastic('dw_alpha', startvalues, 
                                            ranges)
            # The transmission profile of the damping wing.
            self.trans_dampingwing = pymc.Deterministic(
                eval=trans_dampingwing,
                parents = dict(x=self.model_wavelength, 
                               x0=self.dw_lambda_0, 
                               tau0=self.dw_tau0, 
                               xe=self.dw_lambda_e, 
                               alpha=self.dw_alpha),
                name = 'trans_dampingwing',
                doc = trans_dampingwing.__doc__
                )
            # Total spectrum without noise and before convolution.
            self.trueflux0 = (self._trueflux_levels[-1][0] 
                              * self.trans_dampingwing)
        else:
            # No damping wing:
            self.trueflux0 = self._trueflux_levels[-1][0]
        self.trueflux0.keep_trace = False

        if self.convolve_width_pix is not None:
            # The total spectrum (sum of all components) without noise:
            @pymc.deterministic
            def trueflux(tflux=self.trueflux0):
                """The flux without noise: sum of continuum and emission
                lines, convolved with kernel.
                """
                return numpy.convolve(tflux, self.kernel, mode='valid')
        else: 
            self.trueflux = self.trueflux0

        self.trueflux = trueflux
        self.trueflux.keep_trace = False

        # Set up some deterministics needed for priors:
        self.log_peak_ratios = []
        self.log_widths = []
        for (peak, width, vshift) in zip(self.peaks, self.widths, self.vshifts):
            self.log_peak_ratios.append(Log10Deterministic(
                    peak/self.contflux))
            self.__dict__[self.log_peak_ratios[-1].__name__] = \
                self.log_peak_ratios[-1]
            self.log_peak_ratios[-1].keep_trace = False

            self.log_widths.append(Log10Deterministic(
                    width
                    ))
            self.log_widths[-1].keep_trace = False
            self.__dict__[self.log_widths[-1].__name__] = \
                self.log_widths[-1]

            self.__dict__[vshift.__name__] = vshift
        
        # Set up multivariate normal priors on the joint distributions
        # of some parameters:
        if cov_prior_args is not None:
            self.cov_prior_vars = []
            self.cov_prior_potentials = []
            for (i, varnames) in enumerate(cov_prior_args['varnames']):
                # Make a list of the variables.
                self.cov_prior_vars.append([])
                for vn in varnames:
                    if type(vn) is str:
                        # Add variable to the list.
                        self.cov_prior_vars[i].append(self.__dict__[vn])
                    else:
                        # Make a deterministic and add it to the list.
                        va = self.__dict__[vn[0]]
                        op = self.__dict__[vn[1]]
                        vb = self.__dict__[vn[2]]
                        self.cov_prior_vars[i].append(op(va,vb))

                # Make a potential linking these variables.
                self.cov_prior_potentials.append(
                    pymc.Potential(
                        logp=pymc.mv_normal_cov_like,  
                        doc="", 
                        name=cov_prior_args['names'][i], 
                        parents=dict(
                            x = self.cov_prior_vars[i],
                            mu = cov_prior_args['means'][i],
                            C = cov_prior_args['covs'][i]
                            ), 
                        )
                    )

        ### The Likelihood: ###
        # Stochastic representing spectrum with noise. This defines
        # the data likelihood given the model.
        self.flux = pymc.Normal(name='flux', 
                                value=self.scaled_flux[self.goodmask],
                                mu=self.trueflux[self.goodmask], 
                                tau=(self.scaled_error[self.goodmask])**-2, 
                                observed=observed,
                                trace=not(observed))

    def closest_flux(self, wavelength_point):
        """The (scaled) flux in the pixel closest to wavelength_point."""
        closest_index = numpy.argmin(numpy.abs(
                self.wavelength - wavelength_point))
        cf = self.scaled_flux[closest_index]
        return cf

    def continuum_component(self,
                            wavelength, pivot_wavelength, 
                            contindex, contflux):
        """The flux without noise: a continuum component.
        """
        return contflux * (wavelength/pivot_wavelength)**contindex

    def line_component(self,
                       wavelength, center, 
                       width, peak, vshift):
        """The flux without noise: a Gaussian line component.
        """
        shift_factor = 1. + (vshift * 1e5 / cc.c_light_cm_s)
        shifted_center = center * shift_factor
        wavelength_width = center * width * 1e5 / cc.c_light_cm_s
        # Go out only to 4-sigma to speed up calculation on large spectra.
        dev = numpy.square((wavelength - shifted_center) / wavelength_width)
        mask = dev < 16 
        y = numpy.zeros_like(dev)
        y[mask] = numpy.exp(-0.5 * dev[mask])
        return peak * y

    def flux_components(self,
                        wavelength, all_centers, pivot_wavelength, 
                        contindex, contflux, 
                        widths, peaks, vshifts):
        """The flux without noise: continuum and emission line components.
        """
        tf = []
        tf.append(contflux * (wavelength/pivot_wavelength)**contindex)
        for i in xrange(len(all_centers)):
            shift_factor = 1. + (vshifts[i] * 1e5 / cc.c_light_cm_s)
            shifted_center = all_centers[i] * shift_factor
            wavelength_width = (all_centers[i] * widths[i] * 1e5 
                                / cc.c_light_cm_s)
            tf.append(peaks[i] * 
                      numpy.exp(-1 * (wavelength - shifted_center)**2 
                                 / (2 * wavelength_width**2)))
        return tf

    def all_components(self,
                       model_wavelength,
                       all_centers, pivot_wavelength, 
                       contindex, contflux, 
                       widths, peaks, vshifts, 
                       dw_lambda_0=None, 
                       dw_tau0=None,
                       dw_lambda_e=None, 
                       dw_alpha=None,
                       kernel=None,
                       convolve_width_pix=None,
                       ):
        """Continuum, emission lines, damping wing transmission,
        unabsorbed total flux, and absorbed total flux.
        """
        tf = []
        tf.append(contflux * (model_wavelength/pivot_wavelength)**contindex)
        total = tf[0].copy()
        for i in xrange(len(all_centers)):
            shift_factor = 1. + (vshifts[i] * 1e5 / cc.c_light_cm_s)
            shifted_center = all_centers[i] * shift_factor
            wavelength_width = (all_centers[i] * widths[i] * 1e5 
                                      / cc.c_light_cm_s)
            tf.append(peaks[i] * 
                      numpy.exp(-1 * (model_wavelength - shifted_center)**2 
                                 / (2 * wavelength_width**2)))
            total += tf[-1]
        if self.add_dampingwing:
            tf.append(trans_dampingwing(x=model_wavelength, 
                                        x0=dw_lambda_0, 
                                        tau0=dw_tau0, 
                                        xe=dw_lambda_e, 
                                        alpha=dw_alpha))
        else:
            tf.append(numpy.ones_like(model_wavelength))
        tf.append(total)
        # Multiply total flux by transmission.
        tf.append(total * tf[-2]) 
        if convolve_width_pix is not None:
            tf[-1] = scipy.signal.convolve(tf[-1], numpy.atleast_2d(kernel), 
                                           mode='valid')
            tf[-2] = scipy.signal.convolve(tf[-2], numpy.atleast_2d(kernel), 
                                           mode='valid')
        return tf

class PowerGaussMCMC(pymc.MCMC):
    """An MCMC Chain for a PowerGauss model.

    This is a convenient way to set up a pymc.MCMC chain for a
    PowerGauss model.

    Attributes
    ----------

    model : PowerGauss object
        
    """
    def __init__(self, wavelength, error, observed_flux,
                 filename=None, dbname=None, dbdirectory='dbs', 
                 nodb=False, dbclass=None, dbmode='w',
                 epsfactor=0.05, nosave=False, **kwargs):
        """

        Parameters
        ----------
        wavelength, error, observed_flux -- passed to `PowerGauss.__init__`
        filename -- name of a file to store properties
        dbname -- name of a file for the MCMC chain database
        dbdirectory -- directory to store the MCMC chain
        dbclass -- class for the object to store the MCMC chain database
        dbmode -- passed to pymc.MCMC.__init__
        epsfactor -- scales initial proposal distribution width
        kwargs -- additional keyword arguments passed to `PowerGauss.__init__`
        """
        if not hasattr(self, '_tosave'):
            # Record state (currently that should just be the passed
            # arguments) for saving and reloading the object.
            self._tosave = locals().copy()
            del self._tosave['self']
            del self._tosave['kwargs']
            del self._tosave['nosave']
            self._tosave.update(**kwargs)

        # Add wavelength, error, observed_flux to arguments to be
        # passed to PowerGauss.__init.
        kwargs.update(dict(wavelength=wavelength, 
                           error=error, 
                           observed_flux=observed_flux))

        # Set up the probability model.
        self.model = PowerGauss(**kwargs)

        # Now initialize the MCMC object.
        if dbname is not None:
            if dbclass is None:
                dbclass = pymc.database.hdf5
            db = dbclass.load(dbname)
            pymc.MCMC.__init__(self, self.model, db=db, dbmode=dbmode)
        elif nodb:
            pymc.MCMC.__init__(self, self.model, db='ram')
        else:
            if filename is not None:
                self.dbname = os.path.join(
                    dbdirectory,
                    os.path.split(filename)[1] + '.hdf5')
                print self.dbname
            else:
                import time
                timestr = str(int(time.time()*100))
                self.dbname = os.path.join(dbdirectory,
                                               'pymc_'+timestr+'.hdf5')
                print self.dbname
            pymc.MCMC.__init__(self, self.model, db='hdf5', 
                               dbname=self.dbname,
                               dbmode='w')

        # Create starting variance values for the sampler using a
        # fraction of the allowed range for each parameter.
        eps = {}
        varlist = []
        varlist.extend([self.contindex, self.contflux])
        varlist.extend(self.widths)
        varlist.extend(self.peaks)
        varlist.extend(self.vshifts)
        if self.add_dampingwing:
            varlist.extend((self.dw_alpha, self.dw_lambda_e,
                            self.dw_tau0)
                           )
        if hasattr(self, 'scale_factors'):
            varlist.extend(self.scale_factors)
        newvarlist = []
        newvarlist.extend(varlist)
        ranges = self.model.ranges
        for var in varlist:
            if not isinstance(var, pymc.Stochastic):
                # Not a stochastic, remove it from the list.
                newvarlist.remove(var)
                continue
            if var.__name__ in self.model.ranges:
                if str(ranges[var.__name__]).lower() == 'fixed':
                    # Fixed "stochastic", remove it from the list.
                    mcmodel.use_step_method(pymc.NoStepper, var)
                    newvarlist.remove(var)
                    continue
                elif str(ranges[var.__name__][0]).lower() == 'normal':
                    # Normal prior, use the given variance.
                    eps[var] = self.model.ranges[var.__name__][2]**2.
                elif (str(ranges[var.__name__]).lower() == 'oneoverx' or 
                      str(ranges[var.__name__]).lower() == 'oneovernegx'):
                    # OneOverX prior: use magnitude if nonzero, 1 otherwise.
                    if var.value == 0:
                        # Wild guess.
                        eps[var] = 1.0
                    else:
                        # Use the magnitude of the value as the variance.
                        eps[var] = numpy.abs(var.value)
                else:
                    # Assume uniform prior, use the given range.
                    eps[var] = (epsfactor 
                                * (self.model.ranges[var.__name__][1] 
                                   - self.model.ranges[var.__name__][0]))**2.
            else:
                # An Uninformative prior: use magnitude if nonzero, 1 otherwise.
                if var.value == 0:
                    # Wild guess.
                    eps[var] = 1.0
                else:
                    # Use the magnitude of the value as the variance.
                    eps[var] = numpy.abs(var.value)
        self.eps = eps
        self.varlist = newvarlist
        # Set the step method for each Stochastic variable.
        for var in self.varlist:
            self.use_step_method(pymc.Metropolis, 
                                 var, 
                                 proposal_sd=eps[var]**0.5, 
                                 proposal_distribution='Normal')

        # Record the database filename
        self._tosave['dbname'] = self.get_dbfilename()
        
        # Save the information needed to reload this model from a file.
        if filename is not None:
            if not nosave:
                self.save(filename)

    def len(self, chain=-1):
        """The length of the current chain.
        """
        return len(self.db.trace(self.db.trace_names[-1][0], chain=-1)[:])

    def min_deviance_params(self, trim=0):
        """Return a dictionary of parameter values at the minimum deviance.
        
        Finds the point in the trace with the minimum deviance.
        """
        bestind = trim + numpy.argmin(self.db.trace('deviance')[trim:])
        pairfunc = lambda s: (s.__name__, 
                              self.db.trace(s.__name__)[bestind])
        bestdict = dict(map(pairfunc, self.stochastics))
        return bestdict

    def plot_spectra(self, axeslist=None, trim=0, thin=1, chain=-1,
                     component=None, truevalues=None, 
                     vwindow=1.5e4, centers=None,
                     vcolors = ['b', 'r', 'g'], wavelength=None, 
                     extra_observed_flux=None, no_absorption=False,
                     obs_style = dict(), extra_style=dict(),
                     alpha = 0.1, rescale=1.0,
                     **styleargs
                     ):
        if centers is None:
            centers = self.line_centers
        if len(centers) == 0:
            centers = [lambda_LyA_0]
        if wavelength is None:
            wavelength = self.wavelength
            model_wavelength = self.model_wavelength
        else:
            if self.convolve_width_pix is not None:
                kcenter = len(self.kernel) / 2
                lowpoints = (wavelength[0]
                             + (wavelength[0] - wavelength[1])
                             * (kcenter - numpy.arange(kcenter)))
                highpoints = (wavelength[-1]
                              + (wavelength[-1] - wavelength[-2])
                              * (1 + numpy.arange(kcenter)))
                model_wavelength = numpy.hstack(
                    (lowpoints, wavelength, highpoints))
            else:
                model_wavelength = wavelength

        if (len(model_wavelength) == len(self.model_wavelength)
            and numpy.all(model_wavelength == self.model_wavelength)):
            plot_residuals = True
        else:
            plot_residuals = False
        if component is None:
            component = xrange(len(self.all_centers) + 1)
        elif numpy.isscalar(component):
            component = [component,]
            
        import pylab
        if axeslist is None:
            if plot_residuals:
                fig = pylab.figure()
                axes = fig.add_subplot(211)
                resaxes = fig.add_subplot(212, sharex=axes)
                fig2 = pylab.figure()
                vaxes = fig2.add_subplot(211)
                vresaxes = fig2.add_subplot(212, sharex=vaxes)
                axeslist = [axes, resaxes, vaxes, vresaxes]
            else:
                fig = pylab.figure()
                axes = fig.add_subplot(211)
                vaxes = fig.add_subplot(212)
                axeslist = [axes, None, vaxes, None]
        else:
            axes, resaxes, vaxes, vresaxes = axeslist

        passed_obs_style = obs_style
        obs_style = dict(c='k', marker='x', ls='None')
        obs_style.update(passed_obs_style)
        obs_style.update(styleargs)
        axes.plot(self.wavelength,
                  self.observed_flux * rescale, **obs_style)

        passed_extra_style = extra_style
        extra_style = dict(c='r', marker='+', ls='None')
        extra_style.update(passed_extra_style)
        extra_style.update(styleargs)
        if extra_observed_flux is not None:
            axes.plot(wavelength,
                      extra_observed_flux * rescale, **extra_style)

        orig_velocities = wavelength_to_vshift(self.wavelength, centers)
        for v, c in zip (orig_velocities, vcolors):
            vaxes.plot(v, self.observed_flux * rescale,
                       'x', c=c)
        
        model_velocities = wavelength_to_vshift(model_wavelength, centers)
        velocities = wavelength_to_vshift(wavelength, centers)
        varnames = ('all_centers', 'pivot_wavelength', 
                    'contindex', 'contflux', 
                    'widths', 'peaks', 'vshifts',
                    'kernel', 'convolve_width_pix')
        if self.add_dampingwing:
            varnames += ('dw_alpha', 'dw_lambda_e', 'dw_tau0', 'dw_lambda_0')
        params = dict()
        if truevalues is not None:
            trueparams = dict()
        for vname in varnames:
            if vname in self.db.trace_names[0]:
                print "Found %s in db." % vname
                trace = self.db.trace(vname, chain=chain)[:][trim::thin]
                params[vname] = trace.reshape((len(trace), 1))
                if truevalues is not None:
                    trueparams[vname] = truevalues[vname]
            elif (vname in self.__dict__ and 
                  type(self.__dict__[vname]) is list):
                print "Detected %s is a list." % vname
                traces = [
                    self.db.trace(p.__name__, chain=chain)[:][trim::thin]
                    for p in self.__dict__[vname]
                    ]
                traces = [trace.reshape((len(trace), 1)) for trace in traces]
                params[vname] = traces
                if truevalues is not None:
                    truevals = [truevalues[p.__name__]
                                for p in self.__dict__[vname]]
                    trueparams[vname] = truevals
            elif vname in self.__dict__:
                print "Assuming %s can be used directly." % vname
                params[vname] = self.__dict__[vname]
                if truevalues is not None:
                    trueparams[vname] = self.__dict__[vname]
            else:
                print "Parameter %s not found." % vname
        params['model_wavelength'] = model_wavelength

        if no_absorption:
            for i in xrange(len(params['peaks'])):
                params['peaks'][i][params['peaks'][i] < 0] = 0
        if truevalues is not None:
            trueparams['model_wavelength'] = model_wavelength
        tfs = self.model.all_components(**params)
        fluxes = tfs[0:-2]
        total = tfs[-2]
        totalabs = tfs[-1]
        scale = self.scale * rescale
        total *= scale
        totalabs *= scale
        trans_dampingwing = tfs[-3]
        trans_dampingwing *= scale
        if truevalues is not None:
            truetfs = self.model.all_components(**trueparams)
            tfluxes = truetfs[0:-2]
            ttotal = truetfs[-2]
            ttotalabs = truetfs[-1]
            ttotal *= scale
            ttotalabs *= scale
            ttrans_dampingwing = truetfs[-3]
            ttrans_dampingwing *= scale

        colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])

        axes.vlines(x=self.all_centers, 
                    ymin=0, 
                    ymax=scale * numpy.median(params['contflux']))
        
        vcenters = wavelength_to_vshift(self.all_centers, centers)
        for v, c in zip (vcenters, vcolors):
            vaxes.vlines(x=v, 
                         ymin=0, 
                         ymax=scale * numpy.median(params['contflux']),
                         color=c)
            vaxes.axvline(x=0, c='k')
        for i in component:
            tf, c = fluxes[i], colors.next()
            tf *= scale
            spec_style = dict(c=c, ls='-', alpha=alpha)
            spec_style.update(styleargs)
            axes.plot(model_wavelength, tf.transpose(), **spec_style)
            for v, vc in zip (model_velocities, vcolors):
                vspec_style = dict(c=vc, ls='-', alpha=alpha)
                vspec_style.update(styleargs)
                vaxes.plot(v, tf.transpose(), **vspec_style)
            if truevalues is not None:
                ttf = tfluxes[i]
                ttf *= scale
                tspec_style = dict(c=c, ls='--', lw=2.0)
                tspec_style.update(styleargs)
                axes.plot(model_wavelength, ttf, **tspec_style)

        spec_style = dict(c='k', ls='-', alpha=alpha)
        spec_style.update(styleargs)
        axes.plot(wavelength, totalabs.transpose(), **spec_style)
        meantotalabs = numpy.mean(totalabs, axis=0)

        spec_style = dict(c='r', ls=':')
        spec_style.update(styleargs)
        axes.plot(wavelength, meantotalabs, **spec_style)
        if self.add_dampingwing:
            spec_style = dict(c='c', alpha=alpha, ls='-')
            spec_style.update(styleargs)
            axes.plot(wavelength, total.transpose(), **spec_style)
            meantotal = numpy.mean(total, axis=0)
            spec_style = dict(c='b', ls='--')
            spec_style.update(styleargs)
            axes.plot(wavelength, meantotal, **spec_style)
            spec_style = dict(c='g', ls=':', alpha=alpha)
            spec_style.update(styleargs)
            axes.plot(model_wavelength, trans_dampingwing.transpose(),
                      **spec_style)
            spec_style = dict(c='g', ls=':')
            spec_style.update(styleargs)
            meandampingwing = numpy.mean(trans_dampingwing, axis=0)
            axes.plot(model_wavelength, meandampingwing, **spec_style)


        if plot_residuals:
            spec_style = dict(c='c', ls='None', marker='.', alpha=alpha)
            spec_style.update(styleargs)
            resaxes.plot(wavelength, 
                         (self.observed_flux * rescale - totalabs
                          ).transpose(),
                         **spec_style)
            spec_style = dict(c='b', ls='--')
            spec_style.update(styleargs)
            resaxes.plot(wavelength, 
                         self.observed_flux * rescale - meantotalabs, 
                         **spec_style)
            resaxes.axhline(y=0, c='k')
            spec_style = dict(c='b', ls=':')
            spec_style.update(styleargs)
            resaxes.plot(wavelength, 3. * self.error * rescale, **spec_style)
            resaxes.plot(wavelength, -3. * self.error * rescale, **spec_style)

        for v, c in zip (velocities, vcolors):
            spec_style = dict(c=c, ls='-', alpha=alpha)
            spec_style.update(styleargs)
            vaxes.plot(v, totalabs.transpose(), **spec_style)
            if extra_observed_flux is not None:
                spec_style = dict(c=c, ls='None', marker='+', alpha=alpha)
                spec_style.update(styleargs)
                vaxes.plot(v,
                           extra_observed_flux * rescale, **spec_style)
            spec_style = dict(c='k', ls=':')
            spec_style.update(styleargs)
            vaxes.plot(v, meantotalabs, **spec_style)

            if self.add_dampingwing:
                spec_style = dict(c='c', alpha=alpha, ls='-')
                spec_style.update(styleargs)
                vaxes.plot(v, total.transpose(), **spec_style)
                spec_style = dict(c='b', ls='--')
                spec_style.update(styleargs)
                vaxes.plot(v, meantotal, **spec_style)

            if plot_residuals:
                spec_style = dict(c=c, ls='None', marker='.', alpha=alpha)
                spec_style.update(styleargs)
                vresaxes.plot(v, 
                              (self.observed_flux * rescale - totalabs
                               ).transpose(), 
                              **spec_style)
                spec_style = dict(c=c, ls='--')
                spec_style.update(styleargs)
                vresaxes.plot(v, self.observed_flux * rescale - meantotalabs, 
                              **spec_style)
                vresaxes.axvline(x=0, c='k')
                vresaxes.axhline(y=0, c='k')
                spec_style = dict(c=c, ls=':')
                spec_style.update(styleargs)
                vresaxes.plot(v, 3. * self.error * rescale, **spec_style)
                vresaxes.plot(v, -3. * self.error * rescale, **spec_style)
                if truevalues is not None:
                    spec_style = dict(c=c, ls='None', marker='+')
                    spec_style.update(styleargs)
                    resaxes.plot(wavelength, 
                                 self.observed_flux * rescale - ttotal,
                                 **spec_style)

        if truevalues is not None:
            spec_style = dict(c='k', ls='--', lw=2.0)
            spec_style.update(styleargs)
            axes.plot(wavelength, ttotal, **spec_style)
            if plot_residuals:
                spec_style = dict(c='k', ls='None', marker='+')
                spec_style.update(styleargs)
                resaxes.plot(wavelength, 
                             self.observed_flux * rescale - ttotal,
                             **spec_style)


        return axeslist, total

    def save(self, filename):
        """Write state to a pickle file.
        """
        output = open(filename, 'wb')
        pickle.dump(self._tosave, output)
        output.close()

    @classmethod
    def load(cls, filename, **stateoverrides):
        """Load model from a pickle file."""
        inputstream = open(filename, 'r')
        state = pickle.load(inputstream)
        state.update(**stateoverrides)
        inputstream.close()
        return cls(nosave=True, **state)

    def get_dbfilename(self):
        return self.db.dbname

class Log10Deterministic(pymc.Deterministic):
    """A Deterministic whose value is the base-10 logarithm of the
    parameter.
    """
    def __init__(self, x, **kwargs):
        pymc.Deterministic.__init__(
            self,
            lambda x: numpy.log10(x),
            'A Deterministic returning the value of log10(%s)'%(str(x)),
            '(log10_%s)'%(str(x)),
            parents=dict(x=x),
            **kwargs)
        self.keep_trace = False

def trans_dampingwing(x, x0, tau0, xe, alpha):
    """Transmission profile of a Lyman-alpha damping wing.

    Returns
    -------

    exp(-tau0 * (x0 - xe)**-alpha * (x - xe)**alpha) when x > xe

    0 when x <= xe

    Parameters
    ----------

    x : float, ndarray
    x0 : float, ndarray
    tau0 : float, ndarray
    xe : float, ndarray
    alpha : float, ndarray
    
    """
    numpy.seterr(invalid='ignore')
    trans = numpy.exp(-tau0 * (x0 - xe)**-alpha * (x - xe)**alpha)
    trans[x<=xe] = 0.0
    return trans

def check_order(x):
    """Check the order of an array.
    
    Returns 0 if the elements of x are strictly increasing
    (x_i+1 > xi), otherwise returns -inf.
    """
    if numpy.all(numpy.diff(x) > 0):
        return 0
    else:
        return -numpy.inf

def wavelength_to_vshift(wavelength, centers):
    """Convert wavelength shifts into velocity shifts in km/s.

    Parameters
    ----------

    wavelength : float, ndarray
        The shifted wavelength(s).

    centers : list, iterable
        The rest wavelength(s).

    The wavelength and centers parameters must be in the same units,
    but it doesn't matter what units.
    
    Returns
    -------

    A list of velocities (in km/s) of the same length as the centers
    parameter. If wavelength is a numpy array, then each element of
    the returned list will be an array.
    """
    velocities = [(wavelength/cw - 1) * cc.c_light_cm_s / 1e5 
                  for cw in centers]
    return velocities

def make_stochastic(name, startvalues, ranges, default=None):
    """Make a Stochastic as specified in startvalues and ranges
    dictionaries.
    
    ranges[name] can be: 
      ('normal', mu, sigma) --- creates a Normal Stochastic
      (lower, upper) --- a Uniform Stochastic
      None --- an Uninformative Stochastic
      'fixed' --- a Degenerate distribution.
      'oneoverx' --- a OneOverX Stochastic.
      'oneovernegx' --- a OneOverX prior on negative of the parameter

    Optionally, you can provide a default starting value, used only if
    name is not found in the startvalues dictionary.
    """
    if name in startvalues:
        default = startvalues[name]
    if name in ranges:
        if str(ranges[name][0]).lower() == 'normal':
            param = pymc.Normal(name,
                                mu=float(ranges[name][1]),
                                tau=float(ranges[name][2])**-2,
                                value=float(default))
            return param
        elif ranges[name] is None:
            param = pymc.Uninformative(name, value=default)
            return param
        elif str(ranges[name]).lower() == 'fixed':
            param = pymc.Degenerate(name, default)
            return param
        elif str(ranges[name]).lower() == 'oneoverx':
            param = pymc.OneOverX(name, default)
            return param
        elif str(ranges[name]).lower() == 'oneovernegx':
            param = pymc.Stochastic(
                name=name, value=default, doc="", 
                logp=lambda value: pymc.one_over_x_like(-value),
                parents={})
            return param
        elif ranges[name] is not None:
            param = pymc.Uniform(name, value=default,
                                 lower=float(ranges[name][0]),
                                 upper=float(ranges[name][1]))
            return param
    param = pymc.Uninformative(name, value=default)
    return param

def plot_fits(filelist, 
              varname_pairs=None,
              thin=1, trim=0, plotthin=None, plot_spectra=True,
              output_callback=None,
              savename=None,
              minlen=None
              ):
    """Plot pairs of parameters from a set of fits.
    returns value_pairs, which has shape:
    (len(filelist), len(vpairs), 2, len(traces))
    """
    if minlen is None:
        minlen = trim
    if varname_pairs is None:
        m = PowerGaussMCMC.load(filelist[0])
        if m.mcmodel.n_lines == 0:
            varname_pairs = [('dw_alpha', 'dw_lambda_e'),
                             ('dw_alpha', 'dw_tau0'),
                             ('dw_lambda_e', 'dw_tau0'),]
        if m.mcmodel.n_components == 2:
            if m.mcmodel.vshifts[2] is m.mcmodel.vshifts[0]:
                varname_pairs = [('vshift_1', 'vshift_0'),]
            else:
                varname_pairs = [('vshift_2', 'vshift_0'), 
                                 ('vshift_3', 'vshift_1')]
        elif m.mcmodel.n_components == 1:
            if m.mcmodel.vshifts[1] is m.mcmodel.vshifts[0]:
                varname_pairs = [('peak_1', 'peak_0'), 
                                 ((numpy.divide, 'peak_1', 'contflux'), 
                                  (numpy.divide, 'peak_0', 'contflux')),
                                 ('peak_1', 'epeak_4'),
                                 ('peak_0', 'epeak_4'),
                                 ('width_1', 'width_0'),
                                 ('vshift_0', 'evshift_4'),
                                 ('vshift_0', 'peak_0'),
                                 ('vshift_0', 'peak_1'),
                                 ('vshift_0',
                                  (numpy.divide, 'peak_0', 'peak_1')), 
                                 ('vshift_0',
                                  (numpy.divide, 'width_0', 'width_1')), 
                                 ('vshift_0',
                                  (numpy.divide, 'ewidth_4', 'width_1')), 
                                 ('width_1', 'ewidth_4'),
                                 ('width_0', 'ewidth_4'),
                                 ('contflux', 'vshift_1'),
                                 ('contindex', 'vshift_1'),
                                 ]
            else:
                varname_pairs = [('vshift_1', 'vshift_0'), 
                                 ('vshift_1',
                                  (numpy.subtract, 'vshift_0', 'vshift_1')), 
                                 ((numpy.divide, 'peak_1', 'contflux'), 
                                  (numpy.divide, 'peak_0', 'contflux')),
                                 ('vshift_1',
                                  (numpy.divide, 'peak_0', 'peak_1')), 
                                 ('width_1', 'width_0'),
                                 ('vshift_1',
                                  (numpy.divide, 'width_0', 'width_1')), 
                                 ('evshift_4', 
                                  (numpy.subtract, 'vshift_0', 'vshift_1')),
                                 ((numpy.subtract, 'evshift_4', 'vshift_1'),
                                  (numpy.subtract, 'vshift_0', 'vshift_1')),
                                 ('ewidth_4', 
                                  (numpy.subtract, 'vshift_0', 'vshift_1')),
                                 ((numpy.divide, 'ewidth_4', 'width_1'), 
                                  (numpy.subtract, 'vshift_0', 'vshift_1')),
                                 ((numpy.divide, 'epeak_4', 'peak_1'),
                                  (numpy.subtract, 'vshift_0', 'vshift_1')),
                                 ('evshift_3', 'vshift_0'),
                                 ('ewidth_3', 'width_0'),
                                 ('epeak_3', 'peak_0'),
                                 ('contflux', 'vshift_1'),
                                 ('contindex', 'vshift_1'),
                                 ]
        m.mcmodel.db.close()
        del m
    value_dict = defaultdict(list)
    unique_varnames = set()
    for vnames in varname_pairs:
        unique_varnames.update(vnames)
    final_filelist = []
    for filename in filelist:
        label = os.path.split(filename)[-1]
        print filename
        try:
            m = PowerGaussMCMC.load(filename)
        except Exception as e:
            print "Couldn't load file. Got error:", e
            continue
        tracelen = len(m.mcmodel.db.deviance[:])
        print "Found trace of len %i" % tracelen
        if tracelen < minlen:
            print "Not enough iterations in fit! Skipping", filename
            continue
        if plot_spectra:
            pthin = plotthin
            if pthin is None:
                pthin = tracelen/15
            axeslist, total = m.plot_spectra(trim=trim, thin=pthin)
            axes, resaxes, vaxes, vresaxes = axeslist
            axes.set_title(label)
            vaxes.set_title(label)
            axes.figure.set_label(label + '_spectrum')
            vaxes.figure.set_label(label + '_vspectrum')
            xmin = 1150
            xmax = 1275
            axes.set_xlim(xmin,xmax)
            vmin = max((xmin/1215.67 - 1) * cc.c_light_cm_s / 1e5, -1.2e4)
            vmax = min((xmax/1215.67 - 1) * cc.c_light_cm_s / 1e5, 1.2e4)
            vaxes.set_xlim(vmin, vmax)
            scale = numpy.max(total)
            axes.set_ylim(-0.1 * scale, 1.1*scale)
            vaxes.set_ylim(-0.1 * scale, 1.1*scale)
            resaxes.set_ylim(-0.1 * scale, 0.1 * scale)
            vresaxes.set_ylim(-0.1 * scale, 0.1 * scale)

            if output_callback is not None:
                output_callback()

        for vname in list(unique_varnames):
            if type(vname) is tuple:
                v1 = m.mcmodel.db.trace(vname[1])[trim::thin]
                v2 = m.mcmodel.db.trace(vname[2])[trim::thin]
                if not vname[1] in unique_varnames:
                    value_dict[vname[1]].append(v1)
                    unique_varnames.add(vname[1])
                if not vname[2] in unique_varnames:
                    value_dict[vname[2]].append(v2)
                    unique_varnames.add(vname[2])
                value_dict[vname].append(vname[0](v1, v2))
            else:
                value_dict[vname].append(m.mcmodel.db.trace(vname)[trim::thin])
            print "mean", vname, numpy.mean(value_dict[vname][-1])
        final_filelist.append(filename)
        m.mcmodel.db.close()
        del m
    if savename is not None:
        import hdf5_utils
        hdf5_utils.save_hdf5(savename, 
                             dict((str(k), v) 
                                  for k, v in value_dict.iteritems()))

    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    alpha = 0.05
    bins = 10
    histbinslist = [bins,]*2
    import plot2Ddist
    for (ipair, vnames) in enumerate(varname_pairs):
        results = dict(axeslist=None)
        for ifile in xrange(len(final_filelist)):
            results = plot2Ddist.plot2Ddist(
                [value_dict[vnames[0]][ifile], value_dict[vnames[1]][ifile]], 
                labels=vnames,
                axeslist=results['axeslist'], color=colors.next(), 
                scatterstyle=dict(alpha=alpha),
                histbinslist=histbinslist)

    if output_callback is not None:
        output_callback()
    return value_dict



