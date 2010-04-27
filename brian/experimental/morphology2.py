'''
Neuronal morphology module for Brian.
'''
from brian.group import Group
from brian.stdunits import metre
from numpy import *

def loadswc(self,name):
    '''
    Reads a SWC file containing a neuronal morphology.
    
    (TODO)
    Output = tree of branches
    
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
    segments=[] # list of segments
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
        self._segments[segment['n']]=segment # how about a list?
    
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

class Morphology(Group):
    '''
    Neuronal morphology (=tree of branches).
    Initialise with the filename of a swc file.
    
    Each branch has a parent and children (usually two).
    We probably don't need to specify the parent.
    '''
    def __init__(self,filename=None):
        # Morphology is a Group
        self._eqs="""
        diameter : metre
        length : metre
        area : metre**2 # = diameter*length*pi?
        d=diameter
        l=length
        x : metre
        y : metre
        z : metre
        """
        if filename is None: # just a soma
            Group.__init__(self,self._eqs,1)
        else:
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
        # 1) Create the list of segments
        lines=open(filename).read().splitlines()
        segments=[] # list of segments
        branching_points=[] # list of branching points
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
        segments.sort(key=lambda segment:segment['n'])
        Group.__init__(self,self._eqs,len(segments))
        self.diameter=[2*s['radius'] for s in segments]
        self.length=[s['length'] for s in segments]
        self.area=[s['area'] for s in segments]
        self.x=[s['location'][0] for s in segments]
        self.y=[s['location'][1] for s in segments]
        self.z=[s['location'][2] for s in segments]
        self.parent=[s['parent'] for s in segments]
        self.type=[s['type'] for s in segments]
        
        # -- I STOPPED HERE -- CHANGE THIS BELOW --
        # 2) Create the dictionary of branches
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
        