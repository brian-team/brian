'''
Neuronal morphology module for Brian.

Typical values for intrinsic parameters

Cm =0.7 - 1 uF/(cm**2) # specific membrane capacitance
Ri = 70 - 200 ohm*cm # intracellular resistivity
Rm = variable, choose taum=Rm*Cm first # specific membrane resistance (ohm*cm**2)

TODO:
* discretise
* problem with simplify and location
* check if surfacic when applying on a branch
'''
from brian.units import meter,ohm
from brian.stdunits import um,cm,ms,uF,mV
from numpy import sqrt,array,pi,mean
from brian.compartments import *
from brian.membrane_equations import *
from brian import sum

def space_constant(d,Rm,Ri):
    return (d*Rm/Ri)**.5

class Morphology(object):
    '''
    Morphology class
    ----------------
    
    '''
    def __init__(self,name=None,Cm=1*uF/(cm**2)):
        '''
        Cm is the specific membrane capacitance
        '''
        self._segments=[]
        if name is not None:
            self.load(name,Cm)
        
    def load(self,name,Cm=1*uF/(cm**2)):
        '''
        Reads a SWC file containing a neuronal morphology and returns
        a morphology object
        Cm is the specific membrane capacitance.
        
        segments is a dictionary of segments, each segment is a dictionary with keys:
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
        # First create the list of segments
        lines=open(name).read().splitlines()
        segments=[]
        branching_points=[]
        types=['undefined','soma','axon','dendrite','apical','fork','end','custom']
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
                segment=dict(n=n,type=T,location=(x,y,z),radius=R,parent=P)
                if T=='soma':
                    segment['area']=4*pi*segment['radius']**2
                else: # dendrite
                    segment['length']=(sum((array(segment['location'])-array(segments[P]['location']))**2))**.5*meter
                    segment['area']=segment['length']*2*pi*segment['radius']
         
                if (P!=n-1) and (P>-2): # P is a branching point
                    branching_points.append(P)
                
                segments.append(segment)
               
        # Create branches
        # branches = list of dict(begin,end,segments) where segments are segment indexes
        branches=[]
        branch=dict(start=0,segments=[],children=0)
        for segment in segments:
            n=segment['n']
            if segment['n'] in branching_points: # end of branch
                branch['segments'].append(n)
                branch['end']=n
                branches.append(branch)
                branch=dict(start=n+1,segments=[],children=0)
            elif segment['parent']!=n-1: # new branch
                branch['end']=n-1
                branches.append(branch)
                branch=dict(start=n,segments=[n],children=0)
            else:
                branch['segments'].append(n)
        # Last branch
        branch['end']=n
        branches.append(branch)
        
        # Make segment dictionary
        self._segments=dict()
        for segment in segments:
            self._segments[segment['n']]=segment
        
        # Name branches and segments
        # The soma is 'soma'
        self._branches=dict()
        for branch in branches:
            #if branch['type']
            parent=self._segments[branch['start']]['parent']
            if parent in self._segments:
                b=[b for b in branches if parent in b['segments']][0] # parent branch
                if b['name']=='soma':
                    branch['name']=str(b['children'])
                else:
                    branch['name']=b['name']+str(b['children'])
                b['children']+=1
            else:
                branch['name']='soma'
            self._branches[branch['name']]=branch

        self.build_equations(Cm)

    def info(self):
        '''
        Prints information about the morphology
        '''
        print len([s for n,s in self._segments.iteritems() if s['n']==n]),'segments'
        print 'Branches:'
        for name,branch in self._branches.iteritems():
            area=sum([self._segments[s]['area'] for s in branch['segments']])*meter**2
            length=sum([self._segments[s]['length'] for s in branch['segments']\
                        if 'length' in self._segments[s]])*meter
            print name,':',len(branch['segments']),'segments ; area =',area,'; length =',length

    def build_equations(self,Cm=1*uF/(cm**2)):
        '''
        Builds a dictionary of equations.
        Cm is the specific capacitance.
        '''
        self._eqs=dict()
        for n,segment in self._segments.iteritems():
            if segment['n']==n: # otherwise it is a copy
                self._eqs[n]=MembraneEquation(C=Cm*segment['area'])
                self._eqs[n].area=segment['area']
        
    def __getitem__(self,x):
        '''
        Returns a branch or a compartment (equations).
        '''
        if type(x)==type((0,0)): # tuple: compartment
            branch,location=x
            return self.compartment(branch,location)
        else: # single variable: branch
            return self.branch(x)
    
    def index(self,branch,location):
        '''
        Returns the index of the segment at given location on given branch
        '''
        branch=str(branch)
        branch=self._branches[branch]
        if type(location)==type(0): # index
            return branch['start']+location
        else: # position
            x=0*meter
            oldx=x
            for s in branch['segments']:
                x+=self._segments[s]['length']
                if x>location:
                    # The closer one
                    if x-location<location-oldx:
                        return s
                    else:
                        return s-1
                oldx=x
            raise IndexError,'Location not found'
    
    def radius(self,branch,location=None):
        '''
        Returns the radius of a compartment or branch
        '''
        if location is None: # branch: return mean radius
            return mean([self.radius(branch,n) for n in self._branches[branch]['segments']])*meter
        else:
            seg=self._segments[self.index(branch,location)]
            return seg['radius']

    def diameter(self,branch,location=None):
        '''
        Returns the diameter of a compartment or branch
        '''
        return 2*self.radius(branch,location)
            
    def length(self,branch,location=None):
        '''
        Returns the length of a compartment or branch.
        '''
        if location is None: # branch
            l=0*meter
            for n in self._branches[branch]['segments']:
                l+=self.length(branch,n)
            return l
        else:
            return self._segments[self.index(branch,location)]['length']
    
    def area(self,branch,location=None):
        '''
        Returns the area of a compartment or branch.
        '''
        if location is None: # branch
            a=0*meter**2
            for n in self._branches[branch]['segments']:
                a+=self.area(branch,n)
            return a
        else:
            seg=self._segments[self.index(branch,location)]
            if seg['type']=='soma':
                return 4*pi*self.radius(branch,location)**2
            else:
                return self.length(branch,location)*2*pi*self.radius(branch,location)
    
    def branch(self,branch,children=False):
        '''
        Returns a branch.
        
        children = True: children are also returned (subtree)
        '''
        branch=str(branch)
        if children:
            l=[]
            for br in self._branches.iterkeys():
                if (branch=='soma') or br.startswith(branch):
                    l.extend(self.branch(br))
            return l
        else:
            return [self._eqs[n] for n in self._branches[branch]['segments']]
    
    def subtree(self,branch):
        return self.branch(branch,children=True)
    
    def compartment(self,branch,location):
        '''
        Returns the segment at given location on given branch
        '''
        return self._eqs[self.index(branch,location)]
        
    def equations(self,Ri=100*ohm*cm):
        '''
        Returns an Equations object for the whole morphology.
        Ri is the intracellular resistivity.
        Connects all segments together.
        '''
        eqs=Compartments(self._eqs)
        for n,s in self._segments.iteritems():
            if s['n']==n: # otherwise it is a copy
                n0=s['parent']
                if n0 in self._segments:
                    s0=self._segments[n0]
                    if s0['type']=='soma':
                        l=.5*s['length']
                    else:
                        l=.5*(s['length']+s0['length']) # connection length
                    Ra=Ri*l/(pi*.5*(s0['radius']**2+s['radius']**2))
                    eqs.connect(n0,n,Ra)
        return eqs
    
    def simplify(self):
        '''
        Replaces branches by equivalent cylinders.
        Preserves the total area and electrotonic length.
        '''
        for branch in self._branches.itervalues():
            if branch['name']!='soma':
                segs=[self._segments[n] for n in branch['segments']]
                a=sum([seg['area'] for seg in segs])
                l=sum([seg['length']/sqrt(seg['radius']) for seg in segs])
                r=(a/(2*pi*l))**(2./3.)
                l0=(r**.5)*l
                segs[0]['radius']=r*meter
                #segs[0]['length']=l0 # length is the length of connecting segment
                segs[0]['area']=a*meter**2
                self._segments[branch['end']]=segs[0]
                branch['segments']=branch['segments'][0:1]
                # Destroy other segments
                for seg in segs[1:-1]:
                    del self._segments[seg['n']]
    
    def insert_current(self,current,branch,location=None):
        if location is None: # on branch
            if current._prefix=='__current_': # point current
                raise ValueError,"The current should be surfacic"
            branch=morpho.branch(branch)
            for c in branch:
                c+=current
        else:
            c=morpho.compartment(branch,location)
            c+=current

if __name__=='__main__':
    morpho=Morphology('mp_ma_40984_gc2.CNG.swc') # retinal ganglion cell
    morpho.info()
    #print morpho.branch(100)
    #morpho.simplify()
    #morpho.info()
    morpho.insert_current(Current('I=g*(E-vm) : amp/(cm**2)',surfacic=True,
                                  g=1*uF/(cm**2)/(20*ms),E=-75*mV),101,10*um)
    print morpho[101,10*um]
    model=morpho.equations(Ri=70*ohm*cm)
    