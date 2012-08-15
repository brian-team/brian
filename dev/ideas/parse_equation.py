'''
This file demonstrates how parsing equations would look like with pyparsing.
'''
import pprint

from pyparsing import (Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn,
                       Combine, Suppress, alphas, alphanums, restOfLine,
                       LineEnd)

###############################################################################
# Basic Elements
###############################################################################

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
identifier = Word(alphas + "_", alphanums + "_").setResultsName('variable')

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
expression = Combine(OneOrMore((CharsNotIn(':#\n') +
                                Suppress(Optional(LineEnd()))).ignore('#' + restOfLine)),
                     joinString=" ").setResultsName('expression')


# a unit
# TODO: Brian actually allows expressions like amp * ohm -- what exactly do we
#       want to allow here -- any expression returning a Unit?
unit = (Word(alphas, alphanums) | '1').setResultsName('unit')

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt
parameter = Group(identifier + Suppress(':') + unit).setResultsName('parameter')

# Static equation:
# x = 2 * y : volt
static_eq = Group(identifier + Suppress('=') + expression + Suppress(':') +
                  unit).setResultsName('equation')

# Differential equation
# dx/dt = -x / tau : volt
diffop = (Suppress('d') + identifier + Suppress('/') + Suppress('dt'))
annotation = (Suppress('(') + Word(alphas + '-').setResultsName('annotation') +
              Suppress(')'))
diff_eq = Group(diffop + Suppress('=') + expression + Suppress(':') + unit +
                Optional(annotation)).setResultsName('diffeq')

# ignore comments
equation = (parameter | static_eq | diff_eq).ignore('#' + restOfLine)
equations = ZeroOrMore(equation)


###############################################################################
# Examples
###############################################################################
def print_parse_result(parsed):
    for eq in parsed:
        print eq.getName()
        pprint.pprint(dict(eq.items()))

eqs = '''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-\
    g_na*(m*m*m)*h*(v-ENa)-\
    g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpham*(1-m)-betam*m : 1
dn/dt = alphan*(1-n)-betan*n : 1
dh/dt = alphah*(1-h)-betah*h : 1
dge/dt = -ge*(1./taue) : siemens # a comment
dgi/dt = -gi*(1./taui) : siemens
alpham = 0.32*(mV**-1)*(13*mV-v+VT)/ \
    (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
betam = 0.28*(mV**-1)*(v-VT-40*mV)/ \
    (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alphan = 0.032*(mV**-1)*(15*mV-v+VT)/ \
    (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
'''
print_parse_result(equations.parseString(eqs, parseAll=True))

model = '''w:1 # synaptic weight
         dApre/dt=-Apre/taupre : 1 (event-driven)
         dApost/dt=-Apost/taupost : 1 (event-driven) # comment
        '''
print_parse_result(equations.parseString(model, parseAll=True))

eqs = '''
dv/dt = (
         gl*(El-v) + # passive leak
         ge*(Ee-v) + # excitatory synapses
         gi*(Ei-v) - # inhibitory synapses
         g_na*(m*m*m)*h*(v-ENa) # sodium channels-
         g_kd*(n*n*n*n)*(v-EK) # potassium channels
         )/Cm : volt
'''
print_parse_result(equations.parseString(eqs, parseAll=True))
