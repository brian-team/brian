'''
Tutorial 1a: The simplest Brian program
***************************************

Importing the Brian module
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing to do in any Brian program is to load Brian and the names of
its functions and classes. The standard way to do this is to use the Python
``from ... import *`` statement. 
'''
from brian import *
'''
Integrate and Fire model
~~~~~~~~~~~~~~~~~~~~~~~~

The neuron model we will use in this tutorial is the simplest possible
leaky integrate and fire neuron, defined by the differential equation:

    tau dV/dt = -(V-El)

and with a threshold value Vt and reset value Vr.

Parameters
~~~~~~~~~~

Brian has a system for defining physical quantities (quantities with
a physical dimension such as time). The code below illustrates how
to use this system, which (mostly) works just as you'd expect.
'''
tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -60 * mvolt          # resting potential (same as the reset)
'''
The built in standard units in Brian consist of all the fundamental
SI units like second and metre, along with a selection of derived
SI units such as volt, farad, coulomb. All names are lowercase
following the SI standard. In addition, there are scaled versions
of these units using the standard SI prefixes m=1/1000, K=1000, etc. 

Neuron model and equations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to define a neuron model in Brian is to write a list
of the differential equations that define it. For the moment, we'll just
give the simplest possible example, a single differential equation. You
write it in the following form::

    dx/dt = f(x) : unit

where ``x`` is the name of the variable, ``f(x)`` can be any valid Python
expression, and ``unit`` is the physical units of the variable ``x``. In our
case we will write::

    dV/dt = -(V-El)/tau : volt

to define the variable ``V`` with units ``volt``.

To complete the specification of the model, we also define a threshold and reset
value and create a group of 40 neurons with this model.
'''
G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)
'''
The statement creates a new object 'G' which is an instance of the
Brian class :class:`NeuronGroup`, initialised with the values in the
line above and 40 neurons. In Python, you can call a function or initialise
a class using keyword arguments as well as ordered arguments, so
if I defined a function ``f(x,y)`` I could call it as ``f(1,2)`` or
as ``f(y=2,x=1)`` and get the same effect. See the Python tutorial
for more information on this.

For the moment, we leave the neurons in this group unconnected
to each other, each evolves separately from the others.

Simulation
~~~~~~~~~~

Finally, we run the simulation for 1 second of simulated time.
By default, the simulator uses a timestep dt = 0.1 ms.
'''
run(1 * second)
'''
And that's it! To see some of the output of this network, go
to the next part of the tutorial.

Exercise
~~~~~~~~

The units system of Brian is useful for ensuring that everything
is consistent, and that you don't make hard to find mistakes in
your code by using the wrong units. Try changing the units of one
of the parameters and see what happens.

Solution
~~~~~~~~

You should see an error message with a Python traceback (telling
you which functions were being called when the error happened),
ending in a line something like::

    Brian.units.DimensionMismatchError: The differential equations
    are not homogeneous!, dimensions were (m^2 kg s^-3 A^-1)
    (m^2 kg s^-4 A^-1) 
'''
