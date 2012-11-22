'''
Compartmental neurons
'''

from brian.experimental.morphology import Morphology
from brian.stdunits import *
from brian.units import *
from brian.reset import NoReset
from brian.stateupdater import StateUpdater
from brian.inspection import *
from brian.optimiser import *
from itertools import count
from brian.neurongroup import NeuronGroup
from scipy.linalg import solve_banded
from numpy import zeros, ones, isscalar, diag_indices
from numpy.linalg import solve
from brian.clock import guess_clock
from brian.equations import Equations
import functools
import warnings
from math import ceil, log
from scipy import weave
from time import time
import trace
import numpy

try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed: some features in SpatialNeuron will not be available')
    use_sympy = False

__all__ = ['SpatialNeuron', 'CompartmentalNeuron']


class SpatialNeuron(NeuronGroup):
	"""
	Compartmental model with morphology.
	"""
	def __init__(self, morphology=None, model=None, threshold=None, reset=NoReset(),
				refractory=0 * ms, level=0,
				clock=None, unit_checking=True,
				compile=False, freeze=False, implicit=True, Cm=0.9 * uF / cm ** 2, Ri=150 * ohm * cm,
				bc_type = 2, diffeq_nonzero=True):
		clock = guess_clock(clock)
		N = len(morphology) # number of compartments
		
		if isinstance(model, str):
			model = Equations(model, level=level + 1)
		
		model += Equations('''
		v:volt # membrane potential
		''')

		# Process model equations (Im) to extract total conductance and the remaining current
		if use_sympy:
			try:
				membrane_eq=model._string['Im'] # the membrane equation
			except:
				raise TypeError,"The transmembrane current Im must be defined"
			# Check conditional linearity
			ids=get_identifiers(membrane_eq)
			_namespace=dict.fromkeys(ids,1.) # there is a possibility of problems here (division by zero)
			_namespace['v']=AffineFunction()
			eval(membrane_eq,model._namespace['v'],_namespace)
			try:
				eval(membrane_eq,model._namespace['v'],_namespace)
			except: # not linear
				raise TypeError,"The membrane current must be linear with respect to v"
			# Extracts the total conductance from Im, and the remaining current
			z=symbolic_eval(membrane_eq)
			symbol_v=sympy.Symbol('v')
			b=z.subs(symbol_v,0)
			a=-sympy.simplify(z.subs(symbol_v,1)-b)
			gtot_str="_gtot="+str(a)+": siemens/cm**2"
			I0_str="_I0="+str(b)+": amp/cm**2"
			model+=Equations(gtot_str+"\n"+I0_str,level=level+1) # better: explicit insertion with namespace of v
		else:
			raise TypeError,"The Sympy package must be installed for SpatialNeuron"
		
		# Equations for morphology (isn't it a duplicate??)
		eqs_morphology = Equations("""
		diameter : um
		length : um
		x : um
		y : um
		z : um
		area : um**2
		""")
		
		full_model = model + eqs_morphology
		
		NeuronGroup.__init__(self, N, model=full_model, threshold=threshold, reset=reset, refractory=refractory,
							level=level + 1, clock=clock, unit_checking=unit_checking, implicit=implicit)
		self.model_with_diffeq_nonzero = diffeq_nonzero
		self._state_updater = SpatialStateUpdater(self, clock)
		self.Cm = ones(len(self))*Cm
		self.Ri = Ri
		self.bc_type = bc_type #default boundary condition on leaves
		self.bc = ones(len(self)) # boundary conditions on branch points
		self.changed = True
		
		# Insert morphology
		self.morphology = morphology
		self.morphology.compress(diameter=self.diameter, length=self.length, x=self.x, y=self.y, z=self.z, area=self.area)

	def subgroup(self, N): # Subgrouping cannot be done in this way
		raise NotImplementedError

	def __getitem__(self, x):
		'''
		Subgrouping mechanism.
		self['axon'] returns the subtree named "axon".
        
		TODO:
		self[:] returns the full branch.
		'''
		morpho = self.morphology[x]
		N = self[morpho._origin:morpho._origin + len(morpho)]
		N.morphology = morpho
		return N

	def __getattr__(self, x):
		if (x != 'morphology') and ((x in self.morphology._namedkid) or all([c in 'LR123456789' for c in x])): # subtree
			return self[x]
		else:
			return NeuronGroup.__getattr__(self, x)

class SpatialStateUpdater(StateUpdater):
	"""
	State updater for compartmental models.

	"""
	def __init__(self, neuron, clock=None):
		self.eqs = neuron._eqs
		self.neuron = neuron
		self._isprepared = False
		self._state_updater=neuron._state_updater # to update the currents
		self.first_test_gtot=True
		self.callcount=0
		

	def prepare_branch(self, morphology, mid_diameter,ante=0):
		'''
		1) fill neuron.branches and neuron.index with information about the morphology of the neuron
		2) change some wrong values in Aplus and Aminus. Indeed these were correct only if the neuron is a linear cable.
			Knowledge of the morphology gives correct values.
		3) fill neuron.bc (boundary conditions)
		'''
		branch = morphology.branch()
		i=branch._origin
		j= i + len(branch) - 2
		endpoint = j + 1
		self.neuron.index[i:endpoint+1] = self.neuron.BPcount
		children_number = 0
		
		
		#connections between branches
		for x in (morphology.children):#parent of segment n isn't always n-1 at branch points. We need to change Aplus and Aminus
			gc = 2 * msiemens/cm**2
			startpoint = x._origin
			mid_diameter[startpoint] = .5*(self.neuron.diameter[endpoint]+self.neuron.diameter[startpoint])
			self.Aminus[startpoint]=mid_diameter[startpoint]**2/(4*self.neuron.diameter[startpoint]*self.neuron.length[startpoint]**2*self.neuron.Ri)
			if endpoint>0:
				self.Aplus[startpoint]=mid_diameter[startpoint]**2/(4*self.neuron.diameter[endpoint]*self.neuron.length[endpoint]**2*self.neuron.Ri)
			else :
				self.Aplus[startpoint]=gc
				self.Aminus[startpoint]=gc
			children_number+=1
		
		#boundary conditions
		pointType = self.neuron.bc[endpoint]
		hasChild = (children_number>0)
		if (not hasChild) and (pointType == 1): #if the branch point is a leaf of the tree : apply default boundary condition
			self.neuron.bc[endpoint] = self.neuron.bc_type	
		
		
		#extract informations about the branches
		index_ante = self.neuron.index[ante]
		bp = endpoint
		index = self.neuron.BPcount
		self.i_list.append(i)
		self.j_list.append(j)
		self.bp_list.append(bp)
		self.pointType_list.append(max(1,pointType))
		self.pointTypeAnte_list.append(max(1,self.neuron.bc[ante]))
		self.temp[index] = index_ante
		self.id.append(index)
		self.test_list.append((j-i+2)>1)
		for x in xrange(j-i+2):
			self.ante_list.append(ante)
			self.post_list.append(bp)
		if index_ante == 0:
			self.ind0.append(index)
		if pointType==0 :
			self.ind_bctype_0.append(bp)
		
		
		#initialize the parts of the linear systems that will not change
		if (j-i+2)>1:	#j-i+2 = len(branch)
			#initialize ab
			self.ab[0,i:j]= self.Aplus[i:j]
			self.ab[2,i:j]= self.Aminus[i+1:j+1]
			
			#initialize bL
			VL0 = 1 * volt
			self.bL[i] = (- VL0 * self.Aminus[i])
			
			#initialize bR
			VR0 = 1 * volt
			self.bR[j] = (- VR0 * self.Aplus[j+1])
		
		self.neuron.BPcount += 1
		for x in (morphology.children):
			self.prepare_branch(x,mid_diameter,endpoint)
		
	def prepare(self):
		'''
		From Hines 1984 paper, discrete formula is:
		A_plus*V(i+1)-(A_plus+A_minus)*V(i)+A_minus*V(i-1)=Cm/dt*(V(i,t+dt)-V(i,t))+gtot(i)*V(i)-I0(i)
       
		A_plus: i->i+1
		A_minus: i->i-1
		
        This gives the following tridiagonal system:
        A_plus*V(i+1)-(Cm/dt+gtot(i)+A_plus+A_minus)*V(i)+A_minus*V(i-1)=-Cm/dt*V(i,t)-I0(i)
        
        '''
		mid_diameter = zeros(len(self.neuron)) # mid(i) : (i-1) <-> i
		mid_diameter[1:] = .5*(self.neuron.diameter[:-1]+self.neuron.diameter[1:])
		
		self.Aplus = zeros(len(self.neuron)) # A+ i -> j = Aplus(j)
		self.Aminus = zeros(len(self.neuron)) # A- i <- j = Aminus(j)
		self.Aplus[1]= mid_diameter[1]**2/(4*self.neuron.diameter[1]*self.neuron.length[1]**2*self.neuron.Ri)
		self.Aplus[2:]=mid_diameter[2:]**2/(4*self.neuron.diameter[1:-1]*self.neuron.length[1:-1]**2*self.neuron.Ri)
		self.Aminus[1:]=mid_diameter[1:]**2/(4*self.neuron.diameter[1:]*self.neuron.length[1:]**2*self.neuron.Ri) 
		
		self.neuron.index = zeros(len(self.neuron),int) # gives the index of the branch containing the current compartment
		
		self.neuron.BPcount = 0 # number of branch points (or branches). = len(self.neuron.branches)
		
		#the three solutions for V on a branch
		self.vL = zeros((len(self.neuron)),numpy.float64)
		self.vR = zeros((len(self.neuron)),numpy.float64)
		self.d = zeros((len(self.neuron)),numpy.float64)
		
		#matrix and right hand in the tridiagonal systems that we solve to find vL, vR and d.
		self.bL = zeros((len(self.neuron)),numpy.float64)
		self.bR = zeros((len(self.neuron)),numpy.float64)
		self.bd = zeros((len(self.neuron)),numpy.float64)
		self.ab = zeros((3,len(self.neuron)))
		self.ab1_base = zeros(len(self.neuron))
		
		
		self.gtot = zeros(len(self.neuron))
		self.I0 = zeros(len(self.neuron))
		
		self.i_list = [] #the indexes of the first points of the branches in the neuron. len = neuron.BPcount
		self.j_list = [] #the indexes of the last points of the branches in the neuron. len = neuron.BPcount
		self.bp_list = [] #the indexes of the branch points in the neuron. len = neuron.BPcount
		self.pointType_list = [] #boundary condition on bp. len = neuron.BPcount
		self.pointTypeAnte_list = [] #boundary condition on ante. len = neuron.BPcount
		self.index_ante_list1 = [] #index of the parent branch of the current branch. index is in [0,neuron.BPcount]
		self.index_ante_list2 = []
		self.ante_list = [] #the indexes in the neuron of the branch points connected to i, for every compartment. len = len(self.neuron)
		self.post_list = [] #for every compartment, contains the index of the branch point. len = len(self.neuron)
		self.test_list = [] #for each branch : 1 if the branch has more than 3 compartments, else 0
		
		self.id = [] #list of every integer in [0,neuron.BPcount]. used in step to change some values in a matrix
		
		self.temp = zeros(len(self.neuron)) #used to construct index_ante_list0, 1, 2.
		self.ind0 = [] #indexes (in [0,neuron.BPcount]) of the branches connected to compartment 0
		self.ind_bctype_0 = [] #indexes of the branch point with boundary condition 0 (constant V)
		
		# prepare_branch : fill the lists, changes Aplus & Aminus
		self.prepare_branch(self.neuron.morphology, mid_diameter,0)
		
		
		self.index_ante_list1, self.ind1 = numpy.unique(numpy.array(self.temp,int),return_index=True)
		self.ind1 = numpy.sort(self.ind1)
		self.index_ante_list1 = self.temp[self.ind1]
		self.index_ante_list1 = list(self.index_ante_list1)
		self.ind2 = []
		for x in xrange(self.neuron.BPcount):
			self.ind2.append(x)
		self.ind2 = numpy.delete(self.ind2,self.ind1,None) 
		self.ind2 = numpy.setdiff1d(self.ind2, self.ind0, assume_unique=True)
		self.index_ante_list2 = self.temp[self.ind2]
		self.index_ante_list2 = list(self.index_ante_list2)
		
		self.index_ante_list = []
		for idx in xrange(self.neuron.BPcount):
			self.index_ante_list.append(self.temp[idx])
		
		
		# linear system P V = B used to deal with the voltage at branch points and take boundary conditions into account.
		self.P = zeros((self.neuron.BPcount,self.neuron.BPcount))
		self.B = zeros(self.neuron.BPcount)
		self.solution_bp = zeros(self.neuron.BPcount)
		
		#in case of a sealed end, Aminus and Aplus are doubled :
		self.Aminus_bp = self.Aminus[self.bp_list]
		self.Aminus_bp [:] *= self.pointType_list[:]
		self.Aplus_i = self.Aplus[self.i_list]
		self.Aplus_i[:] *= self.pointTypeAnte_list[:]
		
		
	def step(self, neuron):
		
		if self.first_test_gtot and isscalar(neuron._gtot):
			self.first_test_gtot=False
			#neuron._gtot = ones(len(neuron)) * neuron._gtot
			
		self.gtot[:] = neuron._gtot #this compute the value of neuron._gtot.
							#if we call neuron._gtot[1] and then neuron._gtot[2] it does 2 computations
							#here we call it only one time on the whole array. this is much faster
		self.I0 = neuron._I0
		
		#------------------------------------solve tridiagonal systems on the branchs-------------------------
		#ab is the matrix in the tridiagonal systems describing the branches.
		#bd is a right hand in one of these tridiagonal systems.
		if self.neuron.changed : # neuron.changed = True <=> there was a new input somewhere. example : the user does  neuron.I[x] = y
			self.update_ab_base() 
		self.update_ab_gtot()
		self.update_bd()

		self.calculate_vd_vL_vR()
		self.neuron.changed = False
		
		#-----------fill P and B, matrix and right hand used to find the voltage at the branch points-----------------
		
		self.P[:,:] = 0
		self.B[:] = 0
		
		Cm = neuron.Cm[self.bp_list]
		dt = neuron.clock.dt
		gtot = self.gtot[self.bp_list]
		I0 = self.I0[self.bp_list]
		v_bp = neuron.v[self.bp_list]
		vLleft = self.vL[self.i_list]
		vLright = self.vL[self.j_list]
		vRleft = self.vR[self.i_list]
		vRright = self.vR[self.j_list]
		dleft = self.d[self.i_list]
		dright = self.d[self.j_list]
		
		vLleft[:] *= self.test_list[:] #if a branch has less than 3 compartments, this equals 0.
										#thus we can do the same work on every branch point.
		vLright[:] *= self.test_list[:]
		vRleft[:] *= self.test_list[:]
		vRright[:] *= self.test_list[:]
		dleft[:] *= self.test_list[:]
		dright[:] *= self.test_list[:]
		
		self.B[self.index_ante_list1] += - self.Aplus_i[self.ind1[:]] * dleft[self.ind1[:]]
		self.B[self.index_ante_list2] += - self.Aplus_i[self.ind2[:]] * dleft[self.ind2[:]]
		self.B[0] += sum(- self.Aplus_i[self.ind0[:]] * dleft[self.ind0[:]])
		
		self.P[(self.index_ante_list1,self.index_ante_list1)] += self.Aplus_i[self.ind1[:]] * (vLleft[self.ind1[:]] - 1)
		self.P[(self.index_ante_list2,self.index_ante_list2)] += self.Aplus_i[self.ind2[:]] * (vLleft[self.ind2[:]] - 1)
		self.P[0,0] += sum(self.Aplus_i[self.ind0[:]] * (vLleft[self.ind0[:]] - 1))
		
		di = diag_indices(neuron.BPcount)
		
		self.B[:] += - Cm[:]/dt * second * v_bp[:] - I0[:] - self.Aminus_bp[:] * dright[:]
		self.P[di] += - Cm[:]/dt * second - gtot[:] + self.Aminus_bp[:] * (vRright[:] - 1)
		self.P[(self.id,self.index_ante_list)] += self.Aminus_bp[:] *vLright[:]
		self.P[(self.index_ante_list,self.id)] += self.Aplus_i[:] *vRleft[:]
		
		self.P[self.ind_bctype_0,:] = 0
		self.P[(self.ind_bctype_0,self.ind_bctype_0)] = 1
		self.B[self.ind_bctype_0] = neuron.v[self.ind_bctype_0]
		
		#------------------------------------------------------solve PV=B-----------------------------------
		
		self.solution_bp = solve(self.P,self.B)
		neuron.v[self.bp_list] = self.solution_bp[:]
		
		#-------------------------------------------------------update v-------------------------------------
		
		self.finalize_v_global()
		

	def update_ab_base(self): #part of ab that doesn't change if there is no prompt from the operator.
		self.ab1_base[:-1] = (- self.neuron.Cm[:-1] / self.neuron.clock.dt * second - self.Aminus[:-1] - self.Aplus[1:])
		self.ab1_base[-1] = (- self.neuron.Cm[-1] / self.neuron.clock.dt * second - self.Aminus[-1])
		
	def update_ab_gtot(self): #this is called every step. changing part of ab.
		self.ab[1,:] = self.ab1_base[:] - self.neuron._gtot
		
	def update_bd(self): #bd is a right hand side in a tridiagonal system
		self.bd[:] = -self.neuron.Cm[:] / self.neuron.clock.dt * self.neuron.v[:] - self.neuron._I0[:]
	
	def calculate_vd_vL_vR(self):
		for index in xrange(self.neuron.BPcount) :
			if self.test_list[index] :
				i = self.i_list[index]
				j = self.j_list[index]
				self.vL[i:j+1] = solve_banded((1,1),self.ab[:,i:j+1],self.bL[i:j+1],overwrite_ab=False,overwrite_b=False)
				self.vR[i:j+1] = solve_banded((1,1),self.ab[:,i:j+1],self.bR[i:j+1],overwrite_ab=False,overwrite_b=False)
				self.d[i:j+1] = solve_banded((1,1),self.ab[:,i:j+1],self.bd[i:j+1],overwrite_ab=False,overwrite_b=False)
	
	def finalize_v_global(self): #V(x) = V(left) * vL(x) + V(right) * vR(x) + d(x)
		self.neuron.v[:] = self.vL[:] * self.neuron.v[self.ante_list[:]] + self.vR[:] * self.neuron.v[self.post_list[:]] + self.d[:]
		self.neuron.v[self.bp_list] = self.solution_bp[:]
	
	def __call__(self, neuron):
		'''
		Updates the state variables.
		'''
		if not self._isprepared:
			self.prepare()
			self._isprepared=True
			print "state updater prepared"
		self.callcount+=1
		print self.callcount
		#Update I,V
		if neuron.changed :
			self._state_updater.changed = True
		self._state_updater(neuron) #update the currents
		self.step(neuron) #update V
		
	def __len__(self):
		'''
		Number of state variables
		'''
		return len(self.eqs)

CompartmentalNeuron = SpatialNeuron
