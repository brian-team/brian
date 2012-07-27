'''
Compartmental neurons


what the state_updater does every step :
-update the currents
-solve the cable equation on the tree
	*find the solution space on every branch (solve 3 tridiagonal systems)
	*find the voltages of the branch points (a small linear system)
	*determine V with these two informations


TODO :
*remove the neuron.changed attribute. Do some automatic test.
*remove the diffeq_nonzero argument. test it instead.
*test if the state_updater has a "calc" method to update variables like gtot on the gpu.
*solve PV =B on the gpu.
*at the end of a time step, import some variables on the cpu. use the StateMonitor.
*change the tridiagonal solver (useful for neurons with many compartments, >1000). see A Scalable Tridiagonal Solver for GPUs, Kim et al,2011.

explanation on the "type" of a branch point :
1 : normal point in the tree. not a leaf.
2 : a sealed end.
0 : this point has a fixed potential. can be a killed end if V(t=0)=0.
This is an int an is used as such in equations : some values are doubled for a sealed end. ex : Aplus becomes 2*Aplus. so : Aplus <- pointType*Aplus.
'''
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from morphology import Morphology
from brian.stdunits import *
from brian.units import *
from brian.reset import NoReset
from brian.stateupdater import StateUpdater
from gpustateupdater import *
from spatialstateupdater_linear import *
from brian.inspection import *
from brian.optimiser import *
from itertools import count
from brian.neurongroup import NeuronGroup
from numpy import zeros, ones, isscalar
from numpy.linalg import solve
from brian.clock import guess_clock
from brian.equations import Equations
import functools
import warnings
from scipy import weave
from time import time

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
		if self.model_with_diffeq_nonzero :
			self._state_updater = SpatialStateUpdater(self, clock)
		else :
			self._state_updater = LinearSpatialStateUpdater(self, clock)
		self.Cm = ones(len(self))*Cm
		self.Ri = Ri
		self.bc_type = bc_type
		self.bc = ones(len(self))
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
	State updater for compartmental models, using a GPU

	"""
	def __init__(self, neuron, clock=None):
		self.eqs = neuron._eqs
		self.neuron = neuron
		self._isprepared = False
		self._state_updater = GPUExponentialEulerStateUpdater(neuron._eqs,clock=neuron.clock)
		self.first_test_gtot=True
		self.callcount=0
		

	def prepare_branch(self, morphology, mid_diameter,ante=0):
		'''
		1) fill neuron.branches and neuron.index with information about the morphology of the neuron
		2) change some wrong values in Aplus and Aminus. Indeed these were correct only if the neuron is unbranched (linear cable).
			Knowledge of the morphology gives correct values
		3) fill neuron.bc (boundary conditions)
		'''
		branch = morphology.branch()
		i=branch._origin
		j= i + len(branch) - 2
		endpoint = j + 1
		self.neuron.index[i:endpoint+1] = self.neuron.BPcount
		children_number = 0
		
		#connections between branches
		for x in (morphology.children):#parent of segment n isnt always n-1 at branch points. We need to change Aplus and Aminus
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
		if (not hasChild) and (pointType == 1):
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
		if (j-i+2>1):
			self.test_list.append(1)
		else :
			self.test_list.append(0)
		for x in xrange(j-i+2):
			self.ante_list.append(ante)
			self.post_list.append(bp)
			self.ante_list_idx.append(index_ante)
			self.post_list_idx.append(index)
		if index_ante == 0 and index != 0:
			self.ind0.append(index)
		if pointType==0 :
			self.ind_bctype_0.append(bp)
		if self.new_tridiag:
			self.i_list_bis.append(i)
			ii = i
		else:
			ii = self.i_list[-1]
		if j-ii+1>2:
			self.j_list_bis.append(j)
			self.new_tridiag = True
		else :
			self.new_tridiag = False
		
		#initialize the parts of the linear systems that will not change
		if (j-i+2)>1:	#j-i+2 = len(branch)
			#initialize ab
			self.ab0[i:j] = self.Aplus[i+1:j+1]
			self.ab2[i+1:j+1] = self.Aminus[i+1:j+1]
			
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
        
        Boundaries, one simple possibility (sealed ends):
        -(Cm/dt+gtot(n)+A_minus)*V(n)+A_minus*V(n-1)=-Cm/dt*V(n,t)-I0(n)
        A_plus*V(1)-(Cm/dt+gtot(0)+A_plus)*V(0)=-Cm/dt*V(0,t)-I0(0)
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
		
		self.bL = zeros((len(self.neuron)),numpy.float64)
		self.bR = zeros((len(self.neuron)),numpy.float64)
		
		self.ab0 = zeros(len(self.neuron))
		self.ab1 = zeros((len(self.neuron)),numpy.float64)
		self.ab2 = zeros(len(self.neuron))
		self.ab1_base = zeros(len(self.neuron))
		
		
		self.gtot = zeros(len(self.neuron))
		self.I0 = zeros(len(self.neuron))
		
		#some lists of indexes describing the neuron.
		self.i_list = []#the indexes of the first points of the branches in the neuron. len = neuron.BPcount
		self.j_list = [] #the indexes of the last points of the branches in the neuron. len = neuron.BPcount
		self.i_list_bis = []
		self.j_list_bis = [] 
		self.new_tridiag = True
		self.bp_list = [] #the indexes of the branch points in the neuron. len = neuron.BPcount
		self.pointType_list = []#boundary condition on bp. len = neuron.BPcount
		self.pointTypeAnte_list = [] #boundary condition on ante. len = neuron.BPcount
		self.index_ante_list1 = []#index of the parent branch of the current branch. index is in [0,neuron.BPcount]
		self.index_ante_list2 = []
		self.ante_list = [] #the indexes in the neuron of the branch points connected to i, for every compartment. len = len(self.neuron)
		self.post_list = [] #for every compartment, contains the index of the branch point. len = len(self.neuron)
		self.ante_list_idx = []#index in the branches ( in [0,neuron.BPcount])
		self.post_list_idx = []
		self.id = [] #list of every integer in [0,neuron.BPcount]. used in step to change some values in a matrix
		self.test_list = []#for each branch : 1 if the branch has more than 3 compartments, else 0
		self.temp = zeros(len(self.neuron)) #used to construct index_ante_list1, 2.
		self.ind0 = [] #indexes (in [0,neuron.BPcount]) of the branches connected to compartment 0
		self.ind_bctype_0 = [] #indexes of the branch point with boundary condition 0 (constant V)
		
		# prepare_branch : fill neuron.index, neuron.branches, changes Aplus & Aminus
		self.prepare_branch(self.neuron.morphology, mid_diameter,0)
		
		#two arrays used in finalize to know the branch points before and after a compartment :
		self.ante_arr = numpy.array(self.ante_list_idx)
		self.post_arr = numpy.array(self.post_list_idx)
		
		
		self.index_ante_list1, self.ind1 = numpy.unique(self.temp,return_index=True)
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
			
		#in case of a sealed end, Aminus and Aplus are doubled :
		self.Aminus_bp = self.Aminus[self.bp_list]
		self.Aminus_bp [:] *= self.pointType_list[:]
		self.Aplus_i = self.Aplus[self.i_list]
		self.Aplus_i[:] *= self.pointTypeAnte_list[:]
		
		# linear system P V = B used to deal with the voltage at branch points and take boundary conditions into account.
		self.P = zeros((self.neuron.BPcount,self.neuron.BPcount))
		self.B = zeros(self.neuron.BPcount)
		self.solution_bp = zeros(self.neuron.BPcount)
		
		#--------------------------------------------------------GPU------------------------
		n = len(self.neuron)
		
		mod = SourceModule("""
        __global__ void updateAB_gtot(double *ab1, double *ab1_base, double *gtot)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          
          ab1[idx] = ab1_base[idx] - gtot[idx];
          ab1[idx+gridDim.x] = ab1_base[idx] - gtot[idx];
          ab1[idx+2*gridDim.x] = ab1_base[idx] - gtot[idx];
        }
        
        __global__ void updateBD(double *bd, double *Cm, double dt,double *v, double *I0)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          
          bd[idx] = - Cm[idx] / dt * v[idx] - I0[idx];
        }
        
        __global__ void finalizeFun(double *v, double *v_bp, int *ante,int *post, double *b, int m)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          int idx_a = ante[idx];
          int idx_p = post[idx];
          
          v[idx] = b[idx + m] * v_bp[idx_a] + b[idx + 2*m] * v_bp[idx_p] + b[idx]; // vL, vR, d
        }
        
        __global__ void finalizeFunBis(double *v, double *v_bp, int *GPU_data_int)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          int bp = GPU_data_int[4 * idx + 3]; 
          v[bp] = v_bp[idx];
        }
        
        __global__ void initPB(double *P, double *B, int BPcount)
        {
        	int idx = threadIdx.x + blockIdx.x * blockDim.x;
        	int idy = threadIdx.x + blockIdx.y * blockDim.y;
        	
        	P[idx + idy * BPcount] = 0.0;
        	B[idx] = 0.0;
        }
        
        //GPU_data_int is : ante_list, i_list, j_list, bp_list
        //GPU_data_double is : test_list, Aplus, Aminus
        __global__ void fillPB(double *P, double *B, double *b, double *Cm_l, double *gtot_l, int *GPU_data_int,
                        double *GPU_data_double, double *I0_l, double *v_l, int BPcount, double dt, int m)
        { 
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          int idx_ante = GPU_data_int[4 * idx];
          int i = GPU_data_int[4 * idx + 1];
          int j = GPU_data_int[4 * idx + 2];
          int bp = GPU_data_int[4 * idx + 3];
          double test = GPU_data_double[3 * idx];
          double Aplus = GPU_data_double[3 * idx + 1];
          double Aminus = GPU_data_double[3 * idx + 2];
          double Cm = Cm_l[bp];
          double gtot = gtot_l[bp];
          double I0 = I0_l[bp];
          double v_bp = v_l[bp];
          double vLright = b[j + m] * test;
          double vRleft = b[i + 2*m] * test;
          double vRright = b[j + 2*m] * test;
          double dright = b[j] * test;
          
          B[idx] += -Cm/dt * v_bp - I0 -Aminus * dright;
          P[idx * BPcount + idx] += -Cm/dt - gtot + Aminus * (vRright - 1.0);
          P[idx * BPcount + idx_ante] += Aminus * vLright;
          P[idx_ante * BPcount + idx] += Aplus * vRleft;
        }
        
        __global__ void fillPB_bis(double *P, double *B, double *b, int *GPU_data_int, double *GPU_data_double,
                        int BPcount, int *indices, int m)
        { 
          int idx_temp = threadIdx.x + blockIdx.x * blockDim.x;
          int idx = indices[idx_temp];
          int idx_ante = GPU_data_int[4 * idx];
          int i = GPU_data_int[4 * idx + 1];
          int test = GPU_data_double[3 * idx];
          double Aplus = GPU_data_double[3 * idx + 1];
          double vLleft = b[i + m] * test;
          double dleft = b[i] * test;
          
          B[idx_ante] += - Aplus * dleft;
          P[idx_ante * (BPcount + 1)] += Aplus * (vLleft - 1.0);
        }
        
        __global__ void badFillPB_0(double *P, double *B, double *b, int *GPU_data_int, double *GPU_data_double,
                        int *indices, double *Cm_l, double *gtot_l, double *I0_l, double *v_l, int len_indices, int m, double dt)
        { 
          double Cm = Cm_l[0];
          double gtot = gtot_l[0];
          double I0 = I0_l[0];
          double v_0 = v_l[0];
          
          B[0] = - Cm/dt * v_0 - I0;
		  P[0] = - Cm/dt - gtot;
		  
		  int idx;
		  int i;
		  int test;
		  double Aplus;
		  double vLleft;
		  double dleft;
          for (int idx_temp=0;idx_temp<len_indices;idx_temp++)
          {
          	idx = indices[idx_temp];
          	i = GPU_data_int[4 * idx + 1];
          	test = GPU_data_double[3 * idx];
          	Aplus = GPU_data_double[3 * idx + 1];
          	vLleft = b[i + m] * test;
          	dleft = b[i] * test;
          
			P[0] += Aplus * (vLleft - 1.0);
			B[0] += - Aplus * dleft;
		  }
        }
        
        __global__ void resetPB_type0(double *P, double *B, double *v, int *indices,int BPcount)
        {
        	int idx = indices[threadIdx.x] + blockIdx.x * blockDim.x;
        	int idy = threadIdx.x + blockIdx.y * blockDim.y;
        	
        	P[idx + idy * BPcount] = 0.0;
        	P[idx + idx * BPcount] = 1.0;
        	B[idx] = v[idx];
        }
        """)
		
		#------------------------get functions from this module and prepare them.----------------------------
		self.updateAB_gtot = mod.get_function("updateAB_gtot")
		self.updateAB_gtot.prepare(["P","P",'P'],block=(1,1,1))
		
		self.updateBD = mod.get_function("updateBD")
		self.updateBD.prepare(["P","P",'d','P','P'],block=(1,1,1))
		
		self.finalizeFun = mod.get_function("finalizeFun")
		self.finalizeFun.prepare(['P','P','P','P','P','i'],block=(1,1,1))
		
		self.finalizeFunBis = mod.get_function("finalizeFunBis")
		self.finalizeFunBis.prepare(['P','P','P'],block=(1,1,1))
		
		self.initPB = mod.get_function("initPB")
		self.initPB.prepare(['P',"P",'i'],block=(1,1,1))
		
		self.fillPB = mod.get_function("fillPB")
		self.fillPB.prepare(["P","P",'P',"P",'P','P','P','P','P','i','d','i'],block=(1,1,1))
		
		self.fillPB_bis = mod.get_function("fillPB_bis")
		self.fillPB_bis.prepare(["P","P",'P',"P",'P','i','P','i'],block=(1,1,1))
		
		self.badFillPB_0 = mod.get_function("badFillPB_0")
		self.badFillPB_0.prepare(["P","P",'P',"P",'P','P','P','P','P',"P",'i','i','d'],block=(1,1,1))
		
		self.resetPB_type0 = mod.get_function("resetPB_type0")
		self.resetPB_type0.prepare(['P',"P",'P','P','i'],block=(1,1,1))
		
		#---------------------------------export data about the neuron on the GPU--------------------------------------
		
		dtype = numpy.dtype(numpy.float64)
		int_type = numpy.dtype(numpy.int32)
		
		self.P_gpu = cuda.mem_alloc(self.P.size * dtype.itemsize)
		self.B_gpu = cuda.mem_alloc(self.B.size * dtype.itemsize)
		
		GPU_data_int = zeros((self.neuron.BPcount,4))
		GPU_data_double = zeros((self.neuron.BPcount,3))
		GPU_data_int[:,0] = self.index_ante_list[:]
		GPU_data_int[:,1] = self.i_list[:]
		GPU_data_int[:,2] = self.j_list[:]
		GPU_data_int[:,3] = self.bp_list[:]
		GPU_data_double[:,0] = self.test_list[:]
		GPU_data_double[:,1] = self.Aplus_i[:]
		GPU_data_double[:,2] = self.Aminus_bp[:]
		self.GPU_data_int = cuda.mem_alloc(4 * self.neuron.BPcount * int_type.itemsize)
		self.GPU_data_double = cuda.mem_alloc(3 * self.neuron.BPcount * dtype.itemsize)
		cuda.memcpy_htod(self.GPU_data_int,GPU_data_int.astype(numpy.int32))
		cuda.memcpy_htod(self.GPU_data_double,GPU_data_double.astype(numpy.float64))
		
		self.ind0_gpu = cuda.mem_alloc(self.neuron.BPcount * int_type.itemsize)
		cuda.memcpy_htod(self.ind0_gpu,numpy.array(self.ind0,numpy.int32))
		
		self.ind1_gpu = cuda.mem_alloc(self.neuron.BPcount * int_type.itemsize)
		cuda.memcpy_htod(self.ind1_gpu,numpy.array(self.ind1,numpy.int32))
		
		self.ind2_gpu = cuda.mem_alloc(self.neuron.BPcount * int_type.itemsize)
		cuda.memcpy_htod(self.ind2_gpu,numpy.array(self.ind2,numpy.int32))
		
		self.ind_bctype_0_gpu = cuda.mem_alloc(self.neuron.BPcount * int_type.itemsize)
		cuda.memcpy_htod(self.ind_bctype_0_gpu,numpy.array(self.ind_bctype_0,numpy.int32))
		
		self.ab1_base_gpu =  cuda.mem_alloc(n * dtype.itemsize)
		
		self.ab1_gpu =  cuda.mem_alloc(3 * n * dtype.itemsize)
		self.ab1_gpu_ptr = int(self.ab1_gpu)
		
		self.Cm_gpu =  cuda.mem_alloc(n * dtype.itemsize)
		cuda.memcpy_htod(self.Cm_gpu,self.neuron.Cm.astype(numpy.float64))
		
		self.gtot_gpu =  cuda.mem_alloc(n * dtype.itemsize)
		
		self.I0_gpu = cuda.mem_alloc(n * dtype.itemsize)
		
		self.v_gpu = cuda.mem_alloc(n * dtype.itemsize)
		cuda.memcpy_htod(self.v_gpu,self.neuron.v)
		
		ab0 = zeros(3*n).astype(numpy.float64)
		ab0[:n] = self.ab0[:]
		ab0[n:2*n] = self.ab0[:]
		ab0[2*n:3*n] = self.ab0[:]
		
		ab2 = zeros(3*n).astype(numpy.float64)
		ab2[:n] = self.ab2[:]
		ab2[n:2*n] = self.ab2[:]
		ab2[2*n:3*n] = self.ab2[:]
		
		dtype = numpy.dtype(numpy.float64)
		
		self.ab0_gpu =  cuda.mem_alloc(ab0.size * dtype.itemsize)
		self.ab0_gpu_ptr = int(self.ab0_gpu)
		self.ab2_gpu =  cuda.mem_alloc(ab2.size * dtype.itemsize)
		self.ab2_gpu_ptr = int(self.ab2_gpu)
		
		self.bL_gpu =  cuda.mem_alloc(self.bL.size * dtype.itemsize)
		self.bL_gpu_ptr = int(self.bL_gpu)
		self.bR_gpu =  cuda.mem_alloc(self.bR.size * dtype.itemsize)
		self.bR_gpu_ptr = int(self.bR_gpu)
		
		self.b_gpu =  cuda.mem_alloc(3 * self.bR.size * dtype.itemsize)
		self.b_gpu_ptr = int(self.b_gpu) # bd, bL, bR -> vd, vL, vR
		
		cuda.memcpy_htod(self.ab0_gpu, ab0)
		cuda.memcpy_htod(self.ab2_gpu, ab2)
		cuda.memcpy_htod(self.bL_gpu, self.bL)
		cuda.memcpy_htod(self.bR_gpu, self.bR)
		
		self.ante_gpu = cuda.mem_alloc(self.ante_arr.size * self.ante_arr.dtype.itemsize)
		self.post_gpu = cuda.mem_alloc(self.ante_arr.size * self.ante_arr.dtype.itemsize)
		self.v_old_gpu = cuda.mem_alloc(self.neuron.v.size * dtype.itemsize)
		
		cuda.memcpy_htod(self.ante_gpu,self.ante_arr)
		cuda.memcpy_htod(self.post_gpu,self.post_arr)
		
		self.v_branchpoints = zeros(self.neuron.BPcount)
		self.v_bp_gpu = cuda.mem_alloc(self.v_branchpoints.size * dtype.itemsize)
		
	def step(self, neuron):
		"""
		solve the cable equation on the tree
			*find the solution space on every branch (solve 3 tridiagonal systems)
			*find the voltages of the branch points (a small linear system)
			*determine V with these two informations
		"""
		#------------------------------------solve tridiagonal systems on the branchs-------------------------
		#update some variables and matrices
		
		self.calc("_gtot",self.gtot_gpu)
		self.calc("_I0",self.I0_gpu)		
		if self.neuron.changed :
			self.update_ab_base_gpu()
			self.neuron.changed = False
		self.update_ab_gtot_gpu()
		self.update_bd_gpu()
		
		# finds the three solutions that we need to have V. this function solves  ab X = b  for b = bd,bL,bR.
		self.calculate_vd_vL_vR_gpu()
		
		#-----------fill P and B, matrix and right hand used to find the voltage at the branch points-----------------
		# 1) initPB reset P and B. every value is 0.
		# 2) fillPB calculates the coefficients of P and B where there is no lock issue.
		# 3) fillPB_bis does the same on two exclusive lists of indexes in P and B, ind1 and ind2 to avoid lock issues
		# 4) badFillPB_0 ajusts the value in the particular case of 0. this one is not parallel but not big.
		# 5) resetPB_type0 ajusts P and B in case of a branchpoint of type 0 (constant V)
		
		self.initPB.prepared_call((neuron.BPcount, neuron.BPcount),self.P_gpu, self.B_gpu, numpy.int32(neuron.BPcount))
		
		self.fillPB.prepared_call((neuron.BPcount,1),self.P_gpu, self.B_gpu, self.b_gpu,
								self.Cm_gpu, self.gtot_gpu, self.GPU_data_int, self.GPU_data_double, self.I0_gpu, self.v_gpu,
								numpy.int32(neuron.BPcount), self.neuron.clock.dt, numpy.int32(len(neuron)))
		
		self.fillPB_bis.prepared_call((len(self.ind1),1),self.P_gpu, self.B_gpu, self.b_gpu, self.GPU_data_int, self.GPU_data_double,
									numpy.int32(neuron.BPcount), self.ind1_gpu, numpy.int32(len(neuron)))
		if len(self.ind2)>0:
			self.fillPB_bis.prepared_call((len(self.ind2),1),self.P_gpu, self.B_gpu, self.b_gpu, self.GPU_data_int, self.GPU_data_double,
									numpy.int32(neuron.BPcount), self.ind2_gpu, numpy.int32(len(neuron)))
			
		self.badFillPB_0.prepared_call((1,1),self.P_gpu, self.B_gpu, self.b_gpu, self.GPU_data_int, self.GPU_data_double,
									self.ind0_gpu, self.Cm_gpu, self.gtot_gpu, self.I0_gpu, self.v_gpu, numpy.int32(len(self.ind0)),
									numpy.int32(len(neuron)),neuron.clock.dt)#this is not parallel
		
		if len(self.ind_bctype_0):
			self.resetPB_type0.prepared_call((len(self.ind_bctype_0), neuron.BPcount),self.P_gpu, self.B_gpu, self.v_gpu, self.ind_bctype_0_gpu,
										numpy.int32(neuron.BPcount))
		
		#------------------------------------------------------solve PV=B-----------------------------------
		
		cuda.memcpy_dtoh(self.P,self.P_gpu) #copy P and B from the GPU to solve PV=B on the CPU
		cuda.memcpy_dtoh(self.B,self.B_gpu)
		
		self.solution_bp = solve(self.P,self.B) #solve PV=B. better : do it on the GPU, use a sparse solver.
		
		cuda.memcpy_htod(self.v_bp_gpu, self.solution_bp) #copy the results to the GPU
		
		#-------------------------------------------------------update v-------------------------------------
		
		self.finalize_v_gpu() # V(x) = V(left) * vL(x) + V(right) * vR(x) + Vd(x)
		
		cuda.memcpy_dtoh(self.neuron.v,self.v_gpu) #bad : should copy requested variables
	
	def solve_branchpoints(self): #project : solve PV=B on the GPU, using a sparse matrix format and an interative solver. not used.
		BPcount = self.neuron.BPcount
		
		code ="""
            // where to perform the computation
			typedef cusp::device_memory MemorySpace;

			// which floating point type to use
			typedef double ValueType;
			
			// create an empty sparse matrix structure (coo format)
    		cusp::coo_matrix<int, ValueType, MemorySpace> P;
			
			// allocate storage for solution (x) and right hand side (b)
    		cusp::array1d<ValueType, MemorySpace> x(P.num_rows, 0);
    		cusp::array1d<ValueType, MemorySpace> B(P.num_rows, 1);
			
    		//TODO fill P and B
			
    		// set stopping criteria:
    		//  iteration_limit    = 100
    		//  relative_tolerance = 1e-3
    		cusp::default_monitor<ValueType> monitor(B, 100, 1e-3);

    		// set preconditioner (identity)
    		cusp::identity_operator<ValueType, MemorySpace> M(P.num_rows, P.num_rows);

    		// solve the linear system P * x = B with the BiConjugate Gradient Stabilized method
    		cusp::krylov::bicgstab(P, x, B, monitor, M);
    		
        """
		support_code=''
		includepath = ["/usr/local/cuda/include/"]
		libpath = ['/usr/local/cuda/lib/','/usr/lib/']
		
		weave.inline(code,['BPcount'],support_code=support_code,headers = ["<cusp/coo_matrix.h>","<cusp/hyb_matrix.h>","<cusp/krylov/bicgstab.h>","<cuda_runtime_api.h>","<cuda.h>"],
					include_dirs=includepath, library_dirs=libpath, libraries=["cusp"], runtime_library_dirs=libpath, compiler="gcc")
		
	
	
	def calc(self,var,res_gpu): #compute the value of neuron.var and store it in res_gpu. the _state_updater must have a "calc" method.
		self._state_updater.calc(var,res_gpu)
		"""
		#if the _state_updater doesn't have a 'calc' method
		cuda.memcpy_htod(res_gpu,self.neuron.var)
		"""
		
	def update_ab_base_gpu(self): #part of ab that doesn't change if there is no prompt from the operator.
		self.ab1_base[:-1] = (- self.neuron.Cm[:-1] / self.neuron.clock.dt * second - self.Aminus[:-1] - self.Aplus[1:])
		self.ab1_base[-1] = (- self.neuron.Cm[-1] / self.neuron.clock.dt * second - self.Aminus[-1])
		cuda.memcpy_htod(self.ab1_base_gpu,self.ab1_base)
		
	def update_ab_gtot_gpu(self): #this is called every step. changing part of ab.
		n = len(self.neuron)
		self.updateAB_gtot.prepared_call((n,1), self.ab1_gpu, self.ab1_base_gpu, self.gtot_gpu)
		
	def update_bd_gpu(self):
		n = len(self.neuron)
		dt = self.neuron.clock.dt
		
		self.updateBD.prepared_call((n,1), self.b_gpu, self.Cm_gpu, dt, self.v_gpu, self.I0_gpu)
		
	def calculate_vd_vL_vR_gpu(self):
		# b_gpu contains bd,X,X. update it to be bd,bL,bR then solve a tridiagonal system.
		# b_gpu now contains vd,vL,vR
		
		m = len(self.neuron)
		
		ab0_gpu_ptr = self.ab0_gpu_ptr
		ab1_gpu_ptr = self.ab1_gpu_ptr
		ab2_gpu_ptr = self.ab2_gpu_ptr
		bL_gpu_ptr = self.bL_gpu_ptr
		bR_gpu_ptr = self.bR_gpu_ptr
		b_gpu_ptr = self.b_gpu_ptr
		
		code ="""
            cusparseHandle_t handle = 0;
            cusparseCreate(&handle);
            
            double *bL_gpu = (double *)bL_gpu_ptr;
            double *bR_gpu = (double *)bR_gpu_ptr;
            double *b_gpu = (double *)b_gpu_ptr;
            double *ab0_gpu = (double *)ab0_gpu_ptr;
            double *ab1_gpu = (double *)ab1_gpu_ptr;
            double *ab2_gpu = (double *)ab2_gpu_ptr;
            
            //bd is up to date in b_gpu
            cudaMemcpy(b_gpu + m, bL_gpu, m*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(b_gpu + 2*m, bR_gpu, m*sizeof(double), cudaMemcpyDeviceToDevice);
            
            cusparseDgtsvStridedBatch(handle, m, ab2_gpu, ab1_gpu, ab0_gpu, b_gpu, 3, m);
            //now b_gpu contains vd, vL, vR
            cudaDeviceSynchronize();
            
            cusparseDestroy(handle);
        """
		support_code='extern "C" cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle, int m,const double *dl, const double *d,const double *du, double *x,int batchCount, int batchStride);'
		includepath = ["/usr/local/cuda/include/"]
		libpath = ['/usr/local/cuda/lib/','/usr/lib/']
		
		
		weave.inline(code,['ab0_gpu_ptr','ab1_gpu_ptr','ab2_gpu_ptr','bL_gpu_ptr','bR_gpu_ptr','m','b_gpu_ptr'],support_code=support_code,headers = ["<cusparse.h>","<cuda_runtime_api.h>","<cuda.h>"],
					include_dirs=includepath, library_dirs=libpath, libraries=["cusparse"], runtime_library_dirs=libpath, compiler="gcc")
		
	def calculate_vd_vL_vR_gpu_branches(self):
		# an attempt to cut the tridiagonal system in many parts. This is slow and useless.
		m = len(self.neuron)
		systemsCount = len(self.i_list_bis)
		
		i_list = numpy.array(self.i_list_bis)
		j_list = numpy.array(self.j_list_bis)
		
		ab0_gpu_ptr = self.ab0_gpu_ptr
		ab1_gpu_ptr = self.ab1_gpu_ptr
		ab2_gpu_ptr = self.ab2_gpu_ptr
		bL_gpu_ptr = self.bL_gpu_ptr
		bR_gpu_ptr = self.bR_gpu_ptr
		b_gpu_ptr = self.b_gpu_ptr
		
		code ="""
            cusparseHandle_t handle = 0;
            cusparseCreate(&handle);
            
            double *bL_gpu = (double *)bL_gpu_ptr;
            double *bR_gpu = (double *)bR_gpu_ptr;
            
            double *b_gpu = (double *)b_gpu_ptr;
            
            double *ab0_gpu = (double *)ab0_gpu_ptr;
            double *ab1_gpu = (double *)ab1_gpu_ptr;
            double *ab2_gpu = (double *)ab2_gpu_ptr;
            
            //bd is up to date in b_gpu
            cudaMemcpy(b_gpu + m, bL_gpu, m*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(b_gpu + 2*m, bR_gpu, m*sizeof(double), cudaMemcpyDeviceToDevice);
            
            int i;
            int j;
            int n;
            
            for(int idx=0;idx<systemsCount;idx++){
            	i = i_list[idx];
            	j = j_list[idx];
            	n = j-i+1;
            	cusparseDgtsvStridedBatch(handle, n , ab2_gpu+i, ab1_gpu+i, ab0_gpu+i, b_gpu+i, 3, m); //this is an async call
            }
            cudaDeviceSynchronize();
            
            cusparseDestroy(handle);
        """
		support_code='extern "C" cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle, int m,const double *dl, const double *d,const double *du, double *x,int batchCount, int batchStride);'
		includepath = ["/usr/local/cuda/include/"]
		libpath = ['/usr/local/cuda/lib/','/usr/lib/']
		
		
		weave.inline(code,['i_list','j_list','ab0_gpu_ptr','ab1_gpu_ptr','ab2_gpu_ptr','bL_gpu_ptr','bR_gpu_ptr','systemsCount','m','b_gpu_ptr'],support_code=support_code,headers = ["<cusparse.h>","<cuda_runtime_api.h>","<cuda.h>"],
					include_dirs=includepath, library_dirs=libpath, libraries=["cusparse"], runtime_library_dirs=libpath, compiler="gcc")
		
	
	def finalize_v_gpu(self): # V(x) = V(left) * vL(x) + V(right) * vR(x) + Vd(x)
		n = len(self.neuron)
		
		self.finalizeFun.prepared_call((n,1), self.v_gpu, self.v_bp_gpu, self.ante_gpu, self.post_gpu, self.b_gpu, numpy.int32(len(self.neuron)))
		self.finalizeFunBis.prepared_call((self.neuron.BPcount,1), self.v_gpu, self.v_bp_gpu, self.GPU_data_int)
		
		self._state_updater.updateV(self.v_gpu)
	
	
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
		
		if neuron.changed:
			self._state_updater.changed = True
			
		self._state_updater(neuron) #update the currents
		
		self.step(neuron) #integrate the cable equation
		
	def __len__(self):
		'''
		Number of state variables
		'''
		return len(self.eqs)

CompartmentalNeuron = SpatialNeuron