quarum
======

Bayesian quasar spectrum modeling
---------------------------------

This project is in the early development stage.

Usage
-----

Fit a model to a spectrum

 $ python spec_fit.py --single_NV -k=0.870763 -z=2.235563 -c=1 -n=1e4 -p=1 -s=data/SDSS_spectra/spSpec-51993-0542-631.fit -o output/spSpec-51993-0542-631 -b=1e3

Fit a model with two-component Ly-alpha and CIV emission lines (-c=2)

 $ python spec_fit.py --single_NV -k=0.870763 -z=2.235563 -c=2 -n=1e4 -p=1 -s=data/SDSS_spectra/spSpec-51993-0542-631.fit -o output/spSpec-51993-0542-631_c2 -b=1e3

Now load and plot the fit results:

(I suggest you run from an ``ipython --pylab`` session.)

 >>> from quarum import powergauss
 >>> m = powergauss.PowerGaussMCMC.load('output/spSpec-51993-0542-631.mcmc')
 >>> m.plot_spectra(thin=m.len()/10)

.. image:: http://github.com/roban/quarum/raw/master/plots/spSpec-51993-0542-631.png

Load and plot the 2-component fit results:

 >>> m2 = powergauss.PowerGaussMCMC.load('output/spSpec-51993-0542-631_c2.mcmc')
 >>> m2.plot_spectra(thin=m2.len()/10)

.. image:: http://github.com/roban/quarum/raw/master/plots/spSpec-51993-0542-631_c2.png

Plot the joint distributions of widths and shifts for the Ly-alpha line:

 >>> import margplot
 >>> al = margplot.marginal_plot([m.db.width_0[::100], m.db.vshift_0[::100]], color='k')
 >>> margplot.marginal_plot([m2.db.width_0[::100], m2.db.vshift_0[::100]], axeslist=al, color='b')
 >>> margplot.marginal_plot([m2.db.width_1[::100], m2.db.vshift_1[::100]], axeslist=al, color='g')
 >>> al[0].set_xlabel('Width (Ang)')
 >>> al[0].set_ylabel('V_Shift (km/s)')

.. image:: http://github.com/roban/quarum/raw/master/plots/Lya_width_shift.png


If you aren't running from ``ipython --pylab``, show the fits:

 >>> import pylab
 >>> pylab.show()
