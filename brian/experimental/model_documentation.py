import itertools
import string

from sympy import Symbol, Eq, Derivative
from sympy.printing import latex
from sympy.printing.pretty.pretty import PrettyPrinter

from brian import *
from brian.inspection import get_identifiers

__all__ = ['DocumentWriter', 'TextDocumentWriter', 'LaTeXDocumentWriter',
           'document_network', 'labels_from_namespace']

class DocumentWriter(object):
    ''' Base class for documenting a network. Already takes care of assigning
    labels to objects if none are given.
    '''
    
    def document_network(self, net=None, labels=None, sim=True,
                         groups=True, connections=True, operations=True,
                         graph_connections=False):
        '''
        Documents the network `net`, if not network is given, a
        :class:`MagicNetwork` is automatically created and documented, i.e.
        if you are running scripts without any explicit network (you use
        `run(...)` instead of `net.run(...)`), you can simply call
        `writer.document_network()`.
        '''
        if net is None:
            net = MagicNetwork(level=2)
        
        if labels is None:
            labels = {}

        self.do_graph_connections = graph_connections

        letters = string.uppercase
        letter_count = 0
        
        self.short_labels = {}
        self.long_labels = {}
                    
        for group in net.groups:
            if group._owner is group:
                if group in labels:
                    self.short_labels[group], self.long_labels[group] = labels[group]
                else:
                    self.short_labels[group] = letters[letter_count]                
                    self.long_labels[group] = str(group)
                    letter_count += 1
                # process subgroups
                subgroup_counter = 0
                for subgroup_ref in group._subgroup_set:
                    subgroup = subgroup_ref()  # (the saved subgroups are only weak references)
                    if subgroup in labels:
                        self.short_labels[subgroup], self.long_labels[subgroup] = labels[subgroup]
                    else:
                        self.short_labels[subgroup] = self.short_labels[group] + str(subgroup_counter)
                        self.long_labels[subgroup] = str(subgroup)
        
        for other in itertools.chain(net.connections, net.operations):
            if other in labels:
                self.short_labels[other], self.long_labels[other] = labels[other]
            else:
                self.short_labels[other] = letters[letter_count]                
                self.long_labels[other] = str(other)
                letter_count += 1
        
        self.intro()
        
        if sim:
            self.document_sim(net)
        
        if groups:
            self.document_groups(net.groups)
        
        if connections:
            self.document_connections(net.connections)

        if graph_connections:            
            self.graph_connections(net.connections, net.groups)            

        if operations:
            self.document_operations(net.operations)
        
        self.outro()            
    
    def intro(self):
        """
        Is called before any other output function, useful for the start of
        an HTML or LaTeX document.
        """
        pass
    
    def outro(self):
        """
        Is called after all other output function, useful for the end of
        an HTML or LaTeX document.
        """
        pass
    
    def document_sim(self, network):
        """
        Document some general properties of the network, e.g. the timestep used
        in the simulation.
        """
        pass
    
    def document_connections(self, connections):
        """
        Document the connections of the network (including :class:`SpikeMonitor`
        etc. as they are modeled as :class:`Connection` objects).
        """
        pass
    
    def graph_connections(self, connections, groups):
        """
        Draw a graph visualizing the connection structure of the network.
        """
        pass
    
    def document_group(self, group):
        """
        Document a single :class:`NeuronGroup` `group`. Will normally be called
        for every group by :meth:`document_groups`.
        """
        pass
    
    def document_groups(self, groups):
        """
        Document all :class:`NeuronGroup` `groups`. Should normally call
        :meth:`document_group` for every group.
        """
        for group in groups:
            self.document_group(group)

    def document_operations(self, operations):
        """
        Document all :class:`NetworkOperation` `operations` (including
        :class:`StateMonitor` etc. as they are modeled as
        :class:`NetworkOperation` objects). Should normally call
        :meth:`document_operation` for every operation.
        """
        
        for op in operations:
            self.document_operation(op)

    def document_operation(self, operation):
        """
        Document a single :class:`NetworkOperation` `operation`. Should normally
        be called by :meth:`document_operations` for every operation.
        """ 
        pass

    @staticmethod
    def to_sympy_expression(eq_string):
        """
        Simple helper function for converting an Equation string `eq_string` 
        (only the right hand side of an equation) into a `sympy` expression by
        calling `x = Symbol('x')` for every variable `x` in the equation.
        """
        l_namespace = {'Symbol': Symbol}
        # add all variables as sympy symbols to namespace        
        for identifier in get_identifiers(eq_string):
            exec '%s = Symbol("%s")' % (identifier, 
                                        identifier) in {}, l_namespace
        return eval(eq_string, {}, l_namespace)
        
class TextDocumentWriter(DocumentWriter):
    """ 
    Documents the network by printing to stdout, uses `sympy` for formatting
    the equations (including nice Unicode symbols)
    """
    def __init__(self, **kwargs):
        DocumentWriter.__init__(self, **kwargs)
        self.pp = PrettyPrinter(settings={'wrap_line': False})
    
    def document_sim(self, network): 
        clock = guess_clock(network.clock)
        print 'Simulating with dt = ' + str(clock.dt)
    
    def document_connections(self, connections):
        print 'Connections and SpikeMonitors: '
        for con in connections:
            if con.target:
                print '\t%s (%s -> %s)' % (con, self.short_labels[con.source],
                                           self.short_labels[con.target])
        # Monitors are connections without a target
        for con in connections:
            if con.target is None:
                print '\t%s (%s, monitoring %s)' %  (con,
                                                     self.short_labels[con],
                                                     self.short_labels[con.source])

    def document_groups(self, groups):
        if not groups:
            return
        print 'Groups: '
        DocumentWriter.document_groups(self, groups)
                    
    def document_group(self, group):        
        print '\t%s (%s)' % (self.long_labels[group], self.short_labels[group]) 
        
        sympy_time = Symbol('t')
        
        try:
            eqs = group._eqs
            print '\t\tEquations: '
            
            equations = ''
            for varname in eqs._diffeq_names:
                sympy_var = DocumentWriter.to_sympy_expression(varname)
                eq_string = eqs._string[varname]                
                sympy_string = self.pp.doprint(Eq(Derivative(sympy_var, sympy_time),
                                                  DocumentWriter.to_sympy_expression(eq_string)))
                for line in sympy_string.split('\n'):                                        
                    equations +=  '\t\t\t' + line + '\n'
                equations += '\n'
        
            for varname in eqs._eq_names:
                sympy_var = DocumentWriter.to_sympy_expression(varname)
                eq_string = eqs._string[varname]
                sympy_string = self.pp.doprint(Eq(sympy_var,
                                                  DocumentWriter.to_sympy_expression(eq_string)))                                    
                for line in sympy_string.split('\n'):                                        
                    equations +=  '\t\t\t' + line + '\n'
                equations += '\n'
            
            print equations
        except AttributeError:
            pass
        
        if group._subgroup_set:
            print '\t\tSubgroups:'
            for subgroup_ref in group._subgroup_set:
                subgroup = subgroup_ref()
                print '\t\t\t%s (%s)' % (self.long_labels[subgroup],
                                         self.short_labels[subgroup]) 

    def document_operations(self, operations):
        if not operations:
            return
        print 'NetworkOperations and StateMonitors:'
        for op in operations:
            if isinstance(op, StateMonitor):
                print '\t' + str(op) + ' (monitoring: %s)' % self.short_labels[op.P]
            else:
                print '\t' + str(op)


class LaTeXDocumentWriter(DocumentWriter):
    """
    Documents the network by printing LaTeX code to stdout. Prints a full
    document (i.e. including preamble etc.). The resulting LaTeX file needs
    the `amsmath` package.
    
    Note that if you use the `graph_connections=True` option, you additionally
    need the `tikz` and `dot2texi` packages (part of `texlive-core` and
    `texlive-pictures` on Linux systems) and the `dot2tex` tool. To make the 
    conversion from dot code to LaTeX automatically happen on document creation
    you have to pass the `--shell-escape` option to your LaTeX call, e.g.
    `pdflatex --shell-escape network_doc.tex`
    """
    def __init__(self, **kwargs):
        DocumentWriter.__init__(self)
        self.pp = PrettyPrinter(settings={'wrap_line': False})
    
    def intro(self):
        print r'''
\documentclass{article}
\usepackage{amsmath,textcomp}
'''

        if self.do_graph_connections:
            print r'''
\usepackage{dot2texi}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows}
'''
            
        print r'\begin{document}' + '\n'            
        
    def outro(self):
        print r'\end{document}'
    
    def document_sim(self, network): 
        print r'\section{Simulation settings}'
        clock = guess_clock(network.clock)
        print 'Simulating with dt = ' + str(clock.dt)
    
    def document_connections(self, connections):
        if not connections:
            return
        print r'\section{Connections and SpikeMonitors}'
        print r'\begin{itemize}'
        for con in connections:
            if con.target:
                print r'\item %s (%s \textrightarrow{} %s)' % (con, self.short_labels[con.source],
                                           self.short_labels[con.target])
        # Monitors are connections without a target
        for con in connections:
            if con.target is None:
                print '\item %s (monitoring: %s)' %  (con, self.short_labels[con.source])
        print r'\end{itemize}'
    
    def document_groups(self, groups):
        if not groups:
            return
        print r'\section{Groups}'
        for group in groups:
            self.document_group(group)
            
    def document_group(self, group):
        print r'\subsection{%s (%s)}' % (self.long_labels[group], self.short_labels[group]) 
        
        sympy_time = Symbol('t')
        
        try:
            eqs = group._eqs
            print r'\subsubsection*{Equations}'
            
            print LaTeXDocumentWriter.latex_equations(eqs)
            
        except AttributeError:
            pass
        
        if group._subgroup_set:
            print '\t\tSubgroups:'
            for subgroup_ref in group._subgroup_set:
                subgroup = subgroup_ref()
                print '\t\t\t%s (%s)' % (self.long_labels[subgroup],
                                         self.short_labels[subgroup]) 

    def graph_connections(self, connections, groups):
        dot_text = generate_dot_text(connections, groups, self.short_labels)
        print r'\begin{dot2tex}' + '\n' + dot_text + '\n' + r'\end{dot2tex}'
        
    @staticmethod    
    def latex_equation(eq_string):
        """ 
        Helper function to convert the right hand side of an equation string
        `eq_string` to a LaTeX expression (not a full equation in LaTeX terms)
        using `sympy`.
        """   
        # convert equation to latex
        latex_eq = latex(DocumentWriter.to_sympy_expression(eq_string),
                         mode='plain')
    
        return latex_eq
    
    @staticmethod
    def format_equation(lhs, rhs, unit):
        """
        Helper function to convert an equation consisting of two strings for the
        left respectively the right hand side of an equation (`lhs` and `rhs`)
        and a Brian :class:`Unit` `unit` into a LaTeX expression aligning on the
        equality sign for `amsmath`'s `align` environment.
        """  
        return r'%s &= %s &(\text{unit: }%s)\\' % (lhs, rhs, unit) + '\n'
        
    @staticmethod
    def latex_equations(eqs):
        """
        Convert Brian equations (either a -- possibly multi-line -- string or an
        :class:`Equation` object) `eqs` into a LaTeX equation using `amsmath`'s
        `align` environment.
        """
        if isinstance(eqs, str):
            eqs = Equations(eqs)
            
        equations = r'\begin{align*}' + '\n'
        
        for varname in eqs._diffeq_names:
            eq_string = eqs._string[varname]                
            rhs = LaTeXDocumentWriter.latex_equation(eq_string)        
            lhs = r'\frac{\mathrm{d}%s}{\mathrm{d}t}' % (varname)        
            equations += LaTeXDocumentWriter.format_equation(lhs, rhs, eqs._units[varname])
    
        for varname in eqs._eq_names:
            eq_string = eqs._string[varname]                
            rhs = LaTeXDocumentWriter.latex_equation(eq_string)                
            equations += LaTeXDocumentWriter.format_equation(varname, rhs, eqs._units[varname])
    
        equations += r'\end{align*}' + '\n'
    
        return equations


def document_network(net=None, output='text', **kwargs):
    '''
    Convenience method for documenting a network without having to construct
    a :class:`DocumentWriter` object first.
    
    If no network `net` is given, a :class:`MagicNetwork` is automatically
    created and documented. The `output` argument should be either `text` (the
    default) or `latex`. Any further keyword arguments are passed on to the 
    respective :class:`DocumentWriter`.    
    '''
    if output.lower() == 'text':
        writer = TextDocumentWriter()
    elif output.lower() == 'latex':
        writer = LaTeXDocumentWriter()
    else:
        raise ValueError('Unknown output mode "%s"' % output)
    
    writer.document_network(net, **kwargs)

def labels_from_namespace(namespace):
    '''
    Creates a labels dictionary that can be handed over to the
    `document_network` from a given namespace. Would typically be called like
    this:
    `net.document_network(labels=labels_from_namespace(locals()))`
    This allows `document_network` to use the variable names used in the Python
    code as the short labels, making it easier to link the model description
    and the actual code.
    '''
    labels = {}
    
    for name, obj in namespace.iteritems():
        if isinstance(obj, (NeuronGroup, NetworkOperation, Monitor, Connection)):
            labels[obj] = (name, str(obj))
    
    return labels

def generate_dot_text(connections, groups, labels):
    
    dot_description = 'digraph network {\n'    
            
    conn_strings = []
    used_groups = set()
    for c in connections:
        if c.target is None: # a monitor
            conn_strings.append('%s -> %s [style=dotted, label="monitoring"];' % (labels[c],
                                                                                 labels[c.source]))
        else:
            #TODO: Label connections with info about connections (e.g. 1:1)
            #      possible to label arrowhead with target variable?
            conn_strings.append('%s -> %s;' % (labels[c.source],
                                              labels[c.target]))
            used_groups.add(c.target)
        
        used_groups.add(c.source)        
    
    # Assure that every group is shown (or all subgroups of a group are shown)
    for group in groups:
        if group not in used_groups:
            all_subgroups_used = True
            for subgroup in group._subgroup_set:
                if subgroup() not in used_groups:
                    all_subgroups_used = False
                    break            
            if not all_subgroups_used:
                dot_description += '%s;\n' % (label[group])            
    
    dot_description += '\n'.join(conn_strings)
    
    dot_description += '\n}'
    
    return dot_description                
   
if __name__ == '__main__':
    # Test with some example code
    
    defaultclock.dt=0.01*ms

    C=281*pF
    g_L=30*nS
    E_L=-70.6*mV
    V__T=-50.4*mV
    Delta__T=2*mV
    tau_w=40*ms
    a=4*nS
    b=0.08*nA
    I=.8*nA
    V_cut=V__T+5*Delta__T # practical threshold condition
    N=500
    
    eqs="""
    dvm/dt=(g_L*(E_L-vm)+g_L*Delta__T*exp((vm-V__T)/Delta__T)+I-w)/C : volt
    dw/dt=(a*(vm-E_L)-w)/tau_w : amp
    Vr:volt
    """
    
    neuron=NeuronGroup(N,model=eqs,threshold=V_cut,reset="vm=Vr;w+=b")
    neuron.vm=E_L
    neuron.w=a*(neuron.vm-E_L)
    neuron.Vr=linspace(-48.3*mV,-47.7*mV,N) # bifurcation parameter
    
    M=StateSpikeMonitor(neuron,("Vr","w")) # record Vr and w at spike times
    
    # Document the network
    document_network(output='latex', labels=labels_from_namespace(locals()),
                     graph_connections=True)
