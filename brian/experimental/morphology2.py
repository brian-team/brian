'''
Neuronal morphology module for Brian.
'''
from brian.group import Group
from scipy import rand
from numpy import *
from brian.units import meter
from brian.stdunits import um
import warnings
from pylab import figure
try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    warnings.warn('Pylab 0.99.1 is required for 3D plots')

class Morphology(object):
    '''
    Neuronal morphology (=tree of branches).
    '''
    def __init__(self,filename=None):
        if filename is not None:
            self.loadswc(filename)

    def loadswc(self,filename):
        '''
        Reads a SWC file containing a neuronal morphology.
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
        # 1) Create the list of segments, each segment has a list of children
        lines=open(filename).read().splitlines()
        segment=[] # list of segments
        types=['undefined','soma','axon','dendrite','apical','fork','end','custom']
        previousn=-1
        for line in lines:
            if line[0]!='#': # comment
                numbers=line.split()
                n=int(numbers[0])-1
                T=types[int(numbers[1])]
                x=float(numbers[2])*um
                y=float(numbers[3])*um
                z=float(numbers[4])*um
                R=float(numbers[5])*um
                P=int(numbers[6])-1 # 0-based indexing
                if (n!=previousn+1):
                    raise ValueError,"Bad format in file "+filename
                seg=dict(x=x,y=y,z=z,T=T,diameter=2*R,parent=P,children=[])
                location=(x,y,z)
                if T=='soma':
                    seg['area']=4*pi*R**2
                    seg['length']=0*um
                else: # dendrite
                    locationP=(segment[P]['x'],segment[P]['y'],segment[P]['z'])
                    seg['length']=(sum((array(location)-array(locationP))**2))**.5*meter
                    seg['area']=seg['length']*2*pi*R
                if P>=0:
                    segment[P]['children'].append(n)
                segment.append(seg)
                previousn=n
        # We assume that the first segment is the root
        self.create_from_segments(segment)
        
    def create_from_segments(self,segment):
        """
        Recursively create the morphology from a list of segments.
        Each segment has attributes: x,y,z,diameter,area,length (vectors) and children (list).
        It also creates a dictionary of names (_namedkid).
        """
        n=0
        if segment[0]['T']!='soma': # if it's a soma, only one compartment
            while (len(segment[n]['children'])==1) and (segment[n]['T']!='soma'): # Go to the end of the branch
                n+=1
        # End of branch
        branch=segment[:n+1]
        # Set attributes
        self.diameter,self.length,self.area,self.x,self.y,self.z=\
            zip(*[(seg['diameter'],seg['length'],seg['area'],seg['x'],seg['y'],seg['z']) for seg in branch])
        self.diameter,self.length,self.area,self.x,self.y,self.z=array(self.diameter),array(self.length),\
            array(self.area),array(self.x),array(self.y),array(self.z)
        self.type=segment[n]['T'] # normally same type for all compartments in the branch
        # Create children (list)
        self.children=[Morphology().create_from_segments(segment[c:]) for c in segment[n]['children']]
        # Create dictionary of names (enumerates children from number 1)
        self._namedkid={}
        for i,child in enumerate(self.children):
            self._namedkid[i+1]=child
            # Name the child if possible
            if child.type in ['soma','axon','dendrite']:
                if child.type in self._namedkid:
                    self._namedkid[child.type]=None # two children with the same name: erase (see next block)
                else:
                    self._namedkid[child.type]=child
        # Erase useless names
        for k in self._namedkid.keys():
            if self._namedkid[k] is None:
                del self._namedkid[k]
        # If two kids, name them L (left) and R (right)
        if len(self.children)==2:
            self._namedkid['L']=self._namedkid[1]
            self._namedkid['R']=self._namedkid[2]
        return self
    
    def plot(self,axes=None,simple=True,origin=None):
        """
        Plots the morphology in 3D. Units are um.
        axes : the figure axes (new figure if not given)
        simple : if True, the diameter of branches is ignored
        """
        if axes is None: # new figure
            fig=figure()
            axes=Axes3D(fig)
        x,y,z,d=self.x/um,self.y/um,self.z/um,self.diameter/um
        if origin is not None:
            x0,y0,z0=origin
            x=hstack((x0,x))
            y=hstack((y0,y))
            z=hstack((z0,z))
        if len(self.x)==1: # only one compartment: probably just the soma
            axes.plot(x,y,z,"r.",linewidth=d[0])
        else:
            if simple:
                axes.plot(x,y,z,"k")
            else: # linewidth reflects compartment diameter
                for n in range(1,len(x)):
                    axes.plot([x[n-1],x[n]],[y[n-1],y[n]],[z[n-1],z[n]],'k',linewidth=d[n-1])
        for c in self.children:
            c.plot(origin=(x[-1],y[-1],z[-1]),axes=axes,simple=simple)
    
if __name__=='__main__':
    from pylab import show
    morpho=Morphology('mp_ma_40984_gc2.CNG.swc') # retinal ganglion cell
    morpho.plot(simple=True)
    show()
