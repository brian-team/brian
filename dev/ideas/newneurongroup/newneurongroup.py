"""
NewNeuronGroup:
    
    Has multiple states
    
    Each state has:
        Name/ID
        Set of equations
        Set of events
    
    Each event is defined by:
        Condition
        Response
        
    Equations are combined, state updates are performed by code generation,
    linear part of differential equations are grouped together where possible so
    that dot operation can be performed on a contiguous slice, with preference
    to 'main' state.
    
New equation syntax:

statename:
... eqs
statename:
... eqs

First statename is left out, and is called 'main' with id 0.

Equations follow the original Brian form, but you can add additional tags
after each equation to add meta-info:
    (const)
        Variable is constant, can use this info in numerical integration solvers
        
Event syntax:
    condition is straight code
    response is a sequence of statements
    
Special events:
    spike
        condition is threshold
        response is 'event->reset'
    reset
        no condition
        response is given by user 'reset' followed by 'event->main'
    
Have additional special names:

    state -> statename: changes the neuron state
    event -> eventname: immediately calls the response of a given event

Refractoriness:
    Standard form handled like so:
        spike event:
            event->refractory_reset
            state->refractory
        refractory_reset event:
            refractory_until = t+refractory
            event->reset
        reset event:
            provided by user
        end_refractory event:
            condition: t>=refractory_until
            response:
                state->main
    More complicated forms can be implemented
    Additional names:
        refractory : second
        refractory_until : second
        
Higher order differential equations:
    Can be converted to systems of first order differential equations
    What should the syntax be?
        Easy bit is:
            3 * d2x/dt2 + 2 * d2y/dt2 = 6*x
            4 * d2x/dt2 + 1 * d2y/dt2 = 5*y
        But how do you specify units and other properties?
            x in volt, y is const, etc.? But this introduces a different
            syntax to the old one. But it can be backwards compatible, e.g.
                tau dV/dt = Vr-V
                V, Vr in volt
                Vr is const

Other ideas for syntax, not to use strings so heavily, e.g.:

    eqs = Equations(
        # first equations is for state 'main'
        '''
        dV/dt = -(Vr-V)/tau
        dVt/dt = -(Vt0-Vt)/taut
        ''',
        # keywords give state names, here 'refractory'
        refractory='''
        dVt/dt = -(Vt0-Vt)/taut
        ''',
        parameters=['Vr', 'Vt0'],
        units={'V':volt, 'Vt0':volt},
        constants=['Vr', 'Vt0'],
        )
    
This could be backwards compatible by keeping the ':unit' syntax and the
parameter syntax.

Another question is how to share equations between states, e.g. you might want
an adaptation variable to continue evolving during a refractory period even
though the membrane potential stops evolving. You can copy the differential
equation, but it violates the principle of not duplicating code/definitions
(because they might go out of sync).
    - One possibility would be to have a set of shared equations that are always
      active, but I'm not sure there is a good syntax for this and it's not so
      general anyway
          + one syntax would be '(shared)' after the definition
    - You could have a syntax like 'x from main' in the definition of a state,
      to copy the equation of x from state main.
"""
