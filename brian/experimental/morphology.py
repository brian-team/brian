'''
Neuronal morphology module for Brian.

Typical values for intrinsic parameters

Cm =0.7 - 1 uF/(cm**2) # specific membrane capacitance
Ri = 70 - 200 ohm*cm # intracellular resistivity
Rm = variable, choose taum=Rm*Cm first # specific membrane resistance (ohm*cm**2)
'''
from brian.units import meter,ohm
from brian.stdunits import um,cm,ms,uF
from brian.compartments import *
from numpy import sqrt,array,pi

def space_constant(d,Rm,Ri):
    return (d*Rm/Ri)**.5

def read_SWC(name):
    '''
    Reads a SWC file containing a neuronal morphology and returns
    a list of segments, which are dictionaries with keys:
    * parent: index of the parent segment
    * type: type of segment in 'undefined','soma','dendrite','axon','apical','fork','end','custom'
    * length
    * area
    * radius
    * location: (x,y,z) coordinates
    
    Large database at http://neuromorpho.org/neuroMorpho
    
    Information below from http://www.mssm.edu/cnic/swc.html
    
    SWC File Format

    The format of an SWC file is fairly simple. It is a text file consisting of a header with various fields beginning with a # character, and a series of three dimensional points containing an index, radius, type, and connectivity information. The lines in the text file representing points have the following layout. 
    n T x y z R P

    n is an integer label that identifies the current point and increments by one from one line to the next.
    
    T is an integer representing the type of neuronal segment, such as soma, axon, apical dendrite, etc. The standard accepted integer values are given below.
    
        * 0 = undefined
        * 1 = soma
        * 2 = axon
        * 3 = dendrite
        * 4 = apical dendrite
        * 5 = fork point
        * 6 = end point
        * 7 = custom
    
    x, y, z gives the cartesian coordinates of each node.
    
    R is the radius at that node.
    P indicates the parent (the integer label) of the current point or -1 to indicate an origin (soma). 
    '''
    lines=open(name).read().splitlines()
    segments=[]
    types=['undefined','soma','axon','dendrite','apical','fork','end','custom']
    for line in lines:
        if line[0]!='#': # comment
            numbers=line.split()
            n=int(numbers[0]) # ignored
            T=types[int(numbers[1])]
            x=float(numbers[2])*um
            y=float(numbers[3])*um
            z=float(numbers[4])*um
            R=float(numbers[5])*um
            P=int(numbers[6])-1 # 0-based indexing
            segment=dict(type=T,location=(x,y,z),radius=R,parent=P)
            if T=='soma':
                segment['area']=4*pi*segment['radius']**2
            else: # dendrite
                segment['length']=(sum((array(segment['location'])-array(segments[P]['location']))**2))**.5*meter
                segment['area']=segment['length']*2*pi*segment['radius']
            
            segments.append(segment)
    return segments

def discretise_morphology(segments,dx,Rm,Ri):
    '''
    Discretise a morphologic tree with a given space constant precision dx
    (relative to the space constant).
    Rm = specific membrane resistance
    Ri = intracellular resistivity
    '''
    new_segments=[]
    parent=None
    x=0
    x0=0
    # TODO: check indexes and embranchments
    # change area, length etc
    for segment in segments:
        if segment['type']=='soma':
            x0=x
            new_segments.append(segment)
        else:
            x+=segment['length']/space_constant(2*segment['radius'],Rm,Ri)
            if x>=x0+dx:
                x0=x
                new_segments.append(segment)
    return new_segments

if __name__=='__main__':
    segments=read_SWC('P12-DEV175.CNG.swc')
    Cm=1*uF/(cm**2)
    Rm=20*ms/Cm
    Ri=100*ohm*cm
    new_segments=discretise_morphology(segments,.05,Rm,Ri)
    print segments[7]
    print len(segments),len(new_segments)
    print new_segments