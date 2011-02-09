============ B R I A N =============================
A clock-driven simulator for spiking neural networks
====================================================

Version: 1.3.0
Authors:
	Romain Brette
		http://audition.ens.fr/brette/
	Dan Goodman
		http://thesamovar.net/neuroscience
Team:
	Cyrille Rossant (brian.library.modelfitting)
		http://cyrille.rossant.net/
	Bertrand Fontaine (brian.hears)
		http://lpp.psycho.univ-paris5.fr/person.php?name=BertrandF
	Victor Benichoux (brian.hears)
	Boris Gourevitch (brian.hears)
		http://pi314.net/

==== Installation ==========================================================

Requirements: Python (version 2.5-7), the following modules:

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

==== Usage and Documentation ===============================================

See the documentation in the extras download, or online:

	http://www.briansimulator.org/docs

==== Changes ===============================================================

Version 1.2.1 to 1.3.0
----------------------

Major features:

* Added Brian.hears auditory library

Minor features:

* Added new brian.tools.datamanager.DataManager, moved from brian.experimental
* reinit(states=False) will now not reset NeuronGroup state variables to 0.
* modelfitting now has support for refractoriness
* New examples in misc: after_potential, non_reliability, reliability,
  van_rossum_metric, remotecontrolserver, remotecontrolclient
* New experimental.neuromorphic package
* Van Rossum metric added

Improvements:

* SpikeGeneratorGroup is faster for large number of events ("gather" option).
* Speed improvement for pure Python version of sparse matrix preparation
* Speed improvements for spike propagation weave code (50-100% faster).
* Clocks have been changed and should now behave more predictably. In addition,
  you can now specify an order attribute for clocks.
* modelfitting is now based on playdoh 0.3
* modelfitting can now use euler/exp.euler or RK2 integration schemes
* Loading AER data is much faster
* Freezing now uses higher precision (used to only use 12sf now uses 17sf)

Bug fixes:

* Bug in STDP with small values for wmin/wmax fixed (ticket #63)
* Equations/aliases now work correctly in STDP (ticket #56)
* Bug in sparse matrices introduced in scipy 0.8.0 fixed
* Bug in TimedArray when dt keyword is used now fixed (thanks to Adrien
  Wohrer for pointing out the bug).
* Units now work correctly in STDP (ticket #60)
* STDP raises an error if operations are reordered (ticket #57)
* linked_var works with static vars (equations) (ticket #68)
* Changing clock.t during a run won't end the run
* Fixed ticket #66 (unit bug)
* Fixed ticket #64 (bug with freeze)
* Can now run a network with no group
* Exception handling now works properly for C version of circiular spike
  container
* ccircular now builds correctly on linux and 64 bit

Internal changes:

* brian.connection deprecated and replaced by subpackage brian.connections,
  making the code structure much more straightforward and setting up for future
  work on code generation, etc.

Version 1.2.0 to 1.2.1
----------------------

Major features:

* New remote controlling of running Brian scripts via RemoteControlServer
  and RemoteControlClient.
  
Minor features:

* New module tools.io
* weight and sparseness can now both be functions in connect_random
* New StateHistogramMonitor object
* clear now has a new keyword all which allows you to destroy all Brian
  objects regardless of whether or not they would be found by MagicNetwork.
  In addition, garbage collection is called after a clear.
* New method StateMonitor.insert_spikes to have spikes on voltage traces.

Improvements

* The sparseness keyword in connect_random can be a function
* Added 'wmin' to STDP
* You can now access STDP internal variables, e.g. stdp.A_pre, and monitor
  them by doing e.g. StateMonitor(stdp.pre_group, 'A_pre')
* STDP now supports nonlinear equations and parameters
* refractory can now be a vector (see docstring for NeuronGroup) for constant
  resets.
* modelfitting now uses playdoh library
* C++ compiled code is now much faster thanks to adding -ffast-math switch to
  gcc, and there is an option which allows you to set your own
  compiler switches, for example -march=native on gcc 4.2+.
* SpikeGeneratorGroup now has a spiketimes attribute to reset the list of
  spike times.
* StateMonitor now caches values in an array, improving speed for M[i] operation
  and resolving ticket #53

Bug fixes

* Sparse matrices with some versions of scipy
* Weave now works on 64 bit platforms with 64 bit Python
* Fixed bug introduced in 1.2.0 where dense DelayConnection structures would
  not propagate any spikes
* Fixed bug where connect* functions on DelayConnection didn't work with
  subgroups but only with the whole group.
* Fixed bug with linked_var from subgroups not working
* Fixed bug with adding Equations objects together using a shared base equation
  (ticket #9 on the trac)
* unit_checking=False now works (didn't do anything before)
* Fixed bug with using Equations object twice (for two different NeuronGroups)
* Fixed unit checking bug and ZeroDivisionError (ticket #38)
* Fixed rare problems with spikes being lost due to wrong size of SpikeContainer,
  it now dynamically adapts to the number of spikes.
* Fixed ticket #5, ionic_currents did not work with units off
* Fixed ticket #6, Current+MembraneEquation now works
* Fixed bug in modelfitting : the fitness was not computed right with CPUs.
* Fixed bug in modelfitting with random seeds on Unix systems. 
* brian.hears.filtering now works correctly on 64 bit systems

Removed features

* Model has now been removed from Brian (it was deprecated in 1.1).

Version 1.1.3 to 1.2.0
----------------------

Major features:

* Model fitting toolbox (library.modelfitting)

Minor features:

* New real-time ``refresh=`` options added to plotting functions
* Gamma factor in utils.statistics
* New RegularClock object
* Added brian_sample_run function to test installation in place of nose tests

Improvements:

* Speed improvements to monitors and plotting functions
* Sparse matrix support improved, should work with scipy versions up to 0.7.1
* Various improvements to brian.hears (still experimental though)
* Parameters now picklable
* Made Equations picklable

Bug fixes:

* Fixed major bug with subgroups and connections (announced on webpage)
* Fixed major bug with multiple clocks (announced on webpage)
* No warnings with Python 2.6
* Minor bugfix to TimedArray caused by floating point comparisons
* Bugfix: refractory neurons could fire in very extreme circumstances
* Fixed bug with DelayConnection not setting max_delay
* Fixed bug with STP
* Fixed bug with weight=lambda i,j:rand()

New examples:

* New multiprocessing examples
* Added polychronisation example
* Added modelfitting examples
* Added examples of TimedArray and linked_var
* Added examples of using derived classes with Brian
* Realtime plotting example

Version 1.1.2 to 1.1.3
----------------------

* STDP now works with DelayConnection
* Added EventClock
* Added RecentStateMonitor
* Added colormap option to StateMonitor.plot
* Added timed array module, see TimedArray class for details.
* Added optional progress reporting to run()
* New recall() function (converse to forget())
* Added progress reporting module (brian.utils.progressreporting)
* Added SpikeMonitor.spiketimes
* Added developer's guide to docs
* Early version of brian.hears subpackage for auditory modelling
* Various bug fixes

Version 1.1.1 to 1.1.2
----------------------

* Standard functions rand() and randn() can now be used in string resets.
* New forget() function.
* Major bugfix for STP

Version 1.1.0 to 1.1.1
----------------------

* New statistical function: vector_strength
* Bugfix for one line string thresholds/resets

Version 1.0.0 to 1.1.0
----------------------

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
----------------------------------

* 2nd order Runge-Kutta method (use order=2)
* Quantity arrays are disabled (units only for scalars)
* brian_global_config added
* UserComputedConnectionMatrix and UserComputedSparseConnectionMatrix
* SimpleCustomRefractoriness, CustomRefractoriness

Version 1.0.0 RC4 to version 1.0.0 RC5
--------------------------------------

* Bugfix of sparse matrix problems
* Compiled version of spike propagation (much faster for
  networks with lots of spikes)
* Assorted small improvements

Version 1.0.0 RC3 to version 1.0.0 RC4
--------------------------------------

* Added StateSpikeMonitor
* Changed QuantityArray behaviour to work better with numpy, scipy and pylab

Version 1.0.0 RC2 to version 1.0.0 RC3
--------------------------------------

* Small bugfixes

Version 1.0.0 RC1 to version 1.0.0 RC2
--------------------------------------

* Documentation system now much better, using Sphinx, includes
  cross references, index, etc.
* Added VariableReset
* Added run_all_tests()
* numpywrappers module added, but not in global namespace
* Quantity comparison to zero doesn't check units (positivity/negativity)

Version 1.0.0 beta to version 1.0.0 RC1
---------------------------------------

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
