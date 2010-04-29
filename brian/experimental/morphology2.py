'''
Neuronal morphology module for Brian.
'''
from brian.group import Group
from numpy import *
from brian.units import meter
from brian.stdunits import um

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
        # TODO: if it's a soma, only one compartment
        self.create_from_segments(segment)
        
    def create_from_segments(self,segment):
        """
        Recursively create the morphology from a list of segments.
        Each segment has attributes: x,y,z,diameter,area,length,parent,children.
        """
        n=0
        while len(segment[n]['children'])==1:
            n+=1
        # End of branch
        branch=segment[:n+1]
        self.diameter,self.length,self.area,self.x,self.y,self.z=\
            zip(*[(seg['diameter'],seg['length'],seg['area'],seg['x'],seg['y'],seg['z']) for seg in branch])
        # This should be a dictionary
        self.children=[Morphology().create_from_segments(segment[c:]) for c in segment[n]['children']]
        return self
    
if __name__=='__main__':
    morpho=Morphology('mp_ma_40984_gc2.CNG.swc') # retinal ganglion cell
