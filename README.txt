============ B R I A N =============================
A clock-driven simulator for spiking neural networks
====================================================

Version: 1.4.2
Authors:
	Romain Brette
		http://audition.ens.fr/brette/
	Dan Goodman
		http://thesamovar.net/neuroscience
Team:
	Cyrille Rossant
		http://cyrille.rossant.net/
	Bertrand Fontaine
	Victor Benichoux
	Marcel Stimberg
	Jonathan Laudanski

==== Installation ==========================================================

Requirements: Python (version 2.5-7), the following modules:

* numpy (version >=1.4.1)
* scipy (version >= 0.7)
* matplotlib (version >=0.90.1, optional, necessary for plotting )
* sympy (optional, necessary for the "event-based" feature in Synapses)

All operating systems: run 'python setup.py install' from the download folder.

Windows: You can run the installer exe file.

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

Version 1.4.1 to 1.4.2
----------------------
This is a bugfix release that does not add any major features. See the commit
log at http://neuralensemble.org/trac/brian/log/ for details. Note that our
development efforts are now entirely focused on Brian 2
(https://github.com/brian-team/brian2), this will most likely be the last
release in the 1.x series.

Version 1.4.0 to 1.4.1
----------------------

Major features:
* C extensions are compiled by default during installation (with a fallback to
  the Python version if compilation fails) -- this might lead to a considerable
  speedup for users who did not compile those extensions manually before 

Minor features:
* Convenience methods for the Synapses class, allowing to save and load the
  connectivity and to convert the weights into a matrix
* A new openmp option to switch on the use of OpenMP pragmas in generated C code
* Brian hears: Two new models, MiddleEar (filtering by the middle ear) and 
  ZhangSynapse (model of the IHC-AN synapse) 
* Brian hears: New convenience functions to get reasonable axis ticks for
  logarithmic axes
  
Improvements:
* Brian's documentation is now also available under brian.readthedocs.org
* ProgressReporter has context manager support (i.e. can be used in "with"
  statements)
* NeuronGroup and Synapses work with empty model specifications. 
* C version of SpikeContainer is now picklable
 
Bug fixes:
* Synaptic equations referring to variables in the pre- or postsynaptic group
  are never considered as being linear (fixes ticket #83)
* Fix issue with static equations in synaptic models (see
  https://groups.google.com/d/msg/briansupport/-/uqxLK_yoqKUJ )
* Make LinearStateUpdater pickable, even if array B is "NotImplemented".
* Fixed the bug in which the StateSpikeMonitor didn't record variables defined
  with a static equation.
  
* Important bug fixes for brian hears, all users are encouraged to update:
	* Make sure that LinearFilterbank copies it source and therefore not
	  changes it (when not using weave) (fixes ticket #73)
	* Fix some bugs in the TanCarney model
	* Fix shifting multi-channel sounds with fractional=True (fixes ticket #80)

Experimental features:
* A C version of SpikeQueue (used in the Synapses class), which can lead to a
  considerable speedup (see "Advanced concepts/Compiled code" for instructions
  how to use it).
* Delays can be specified as a parameters of the Synapses model and then be
  changed dynamically.

Version 1.3.1 to 1.4.0
----------------------

Major features:

* New Synapses class (plasticity, gap junctions, nonlinear synapses, etc)

Minor features:

* New AERSpikeMonitor class
* Several updates to library.electrophysiology

Improvements:

* Units should work better with static code analysers now
* Added Network.remove
* SpikeMonitor has a new .it attribute (returns pair i, t of arrays of spike times)
* Many new examples

Bug fixes:

* Assigning to a static variable (equation) now raise an error
* Fixed issues for TimedArrays with explicitly set times (fixes ticket #81)
* Fixed bug, repr and str didn't work for Sound
* Fixed bug where tone(array_of_frequencies, ...)
* Fixed SparseConnectionMatrix bug suggested by Owen Mackwood
* Fixed bug in Parameters reported by Jimmy Bonaiuto
* Fixed bug with contained_objects reported by Oleg Sinyavskiy
* Units __repr__ and __str__ fixes
* Sound.spectrum, Sound.pinknoise, brownnoise
* t wasn't available in StringReset and PythonThreshold

Deprecated or removed features:

* MultipleSpikeGeneratorGroup
* experimental.coincidence_detection

Experimental features:

* Generating model documentation automatically (experimental.model_documentation) 

Version 1.3.0 to 1.3.1
----------------------

Minor features:

* New PoissonInput class
* New auditory model: TanCarney (brian.hears)
* Many more examples from papers
* New electrode compensation module (in library.electrophysiology)
* New trace analysis module (in library.electrophysiology)
* Added new brian.tools.taskfarm.run_tasks function to use multiple CPUs to
  perform multiple runs of a simulation and save results to a DataManager,
  with an optional GUI interface.
* Added FractionalDelay filterbank to brian.hears, fractional itds to
  HeadlessDatabase and fractional shifts to Sound.shifted.
* Added vowel function to brian.hears for creating artificial vowel sounds
* New spike_triggered_average function
* Added maxlevel and atmaxlevel to Sound
* New IRNS/IRNO noise functions

Improvements:

* SpikeGeneratorGroup is much faster.
* Added RemoteControlClient.set(var, name) to allow sending data to the server
  from the client (previously you could only receive data from the server but
  not send it, except in string form).
* Monitors do not process empty spike arrays when there have not been any 
  spikes, increases speed for monitored networks with sparse firing (#78) 
* Various speed optimisations

Bug fixes:

* Fixed bug with frozen equations and time variable in equations
* Fixed bug with loading sounds using Sound('filename.wav')
* SpikeMonitor now clears spiketimes correctly on reinit (#75)
* MultiConnection now propagates reinit (important for monitors) (#76)
* Fixed bug in realtime plotting
* Fixed various bugs in Sound
* Fixed bugs in STDP
* Bow propagates spikes only if spikes exist (#78)

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
