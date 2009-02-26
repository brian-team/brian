============ B R I A N =============================
A clock-driven simulator for spiking neural networks
====================================================

Version: 1.1.2
Authors:
	Romain Brette
		http://www.di.ens.fr/~brette/index.html
	Dan Goodman
		http://thesamovar.net/neuroscience

==== Installation ==========================================================

Requirements: Python (version 2.5), the following modules:

* numpy
* scipy (preferably 0.7 or later)
* pylab

Windows: run the installer exe file

Others: run 'python setup.py install' from the download folder.

==== Extras ================================================================

Included in the extras download are:

docs
	Documentation for Brian including tutorials.

examples
	Examples of using Brian, these serve as supplementary documentation.
	
tutorials
	Fully worked through tutorials on using Brian. These can be read
	through in the documentation too.	

benchmarks
	Currently includes Scilab and Matlab versions of the CUBA and COBA
	examples, and a C++ version of the CUBA example.

==== Usage and Documentation ===============================================

See the documentation in the extras download, or online:

	http://www.briansimulator.org/docs

==== Changes ===============================================================

Version 1.1.1 to 1.1.2:

* Standard functions rand() and randn() can now be used in string resets.
* New forget() function.
* Major bugfix for STP

Version 1.1.0 to 1.1.1:

* New statistical function: vector_strength
* Bugfix for one line string thresholds/resets

Version 1.0.0 to 1.1.0:

* STDP
* Short-term plasticity (Tsodyks-Markram model)
* New DelayConnection for heterogeneous delays
* New code for Connections, including new 'dynamic' connection matrix type
* Reset and threshold can be specified with strings (Python expressions)
* Much improved documentation
* clear() function added for ipython users
* Simplified initialisation of Connection objects
* Optional unit checking in NeuronGroup
* Spike train statistics (utils.statistics)
* Miscellaneous optimisations
* New MultiStateMonitor class
* New Group, MultiGroup objects (for convenience of people writing extensions mostly)
* Improved contained_objects protocol with ObjectContainer class in brian.base
* UserComputed* classes removed for this version (they will return in another form).

Version 1.0.0 RC5 to version 1.0.0

* 2nd order Runge-Kutta method (use order=2)
* Quantity arrays are disabled (units only for scalars)
* brian_global_config added
* UserComputedConnectionMatrix and UserComputedSparseConnectionMatrix
* SimpleCustomRefractoriness, CustomRefractoriness

Version 1.0.0 RC4 to version 1.0.0 RC5:

* Bugfix of sparse matrix problems
* Compiled version of spike propagation (much faster for
  networks with lots of spikes)
* Assorted small improvements

Version 1.0.0 RC3 to version 1.0.0 RC4:

* Added StateSpikeMonitor
* Changed QuantityArray behaviour to work better with numpy, scipy and pylab

Version 1.0.0 RC2 to version 1.0.0 RC3:

* Small bugfixes

Version 1.0.0 RC1 to version 1.0.0 RC2:

* Documentation system now much better, using Sphinx, includes
  cross references, index, etc.
* Added VariableReset
* Added run_all_tests()
* numpywrappers module added, but not in global namespace
* Quantity comparison to zero doesn't check units (positivity/negativity)

Version 1.0.0 beta to version 1.0.0 RC1:

* Connection: connect_full allows a functional weight argument (like connect_random)
* Short-term plasticity:
  In Connection: 'modulation' argument allows modulating weights by a state
  variable from the source group (see examples).
* HomogeneousCorrelatedSpikeTrains: input spike trains with exponential correlations.
* Network.stop(): stops the simulation (can be called by a user script)
* PopulationRateMonitor: smooth_rate method
* Optimisation of Euler code: use compile=True when initialising NeuronGroup
* More examples
* Pickling now works (saving and loading states)
* dot(a,b) now works correctly with qarray's with homogeneous units
* Parallel simulations using Parallel Python (independent simulations only)
* Example of html inferfaces to Brian scripts using CherryPy
* Time dependence in equations (see phase_locking example)
* SpikeCounter and PopulationSpikeCounter