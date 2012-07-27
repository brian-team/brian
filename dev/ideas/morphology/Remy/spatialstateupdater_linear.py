import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from morphology import Morphology
from brian.stdunits import *
from brian.units import *
from brian.reset import NoReset
from brian.stateupdater import StateUpdater
from gpustateupdater import *
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

try:
    import sympy
    use_sympy = True
except:
    warnings.warn('sympy not installed: some features in SpatialNeuron will not be available')
    use_sympy = False

__all__ = ['LinearSpatialStateUpdater']

class LinearSpatialStateUpdater(StateUpdater):
	"""
	State updater for compartmental models.

	"""
	def __init__(self, neuron, clock=None):
		self.eqs = neuron._eqs
		self.neuron = neuron
		self._isprepared = False
		self.first_step = True
		self._state_updater=neuron._state_updater # to update the currents
		self.first_test_gtot=True
		self.callcount=0
		

	def prepare_branch(self, morphology, mid_diameter,ante=0):
		'''
		1) fill neuron.branches and neuron.index with information about the morphology of the neuron
		2) change some wrong values in Aplus and Aminus. Indeed these were correct only if the neuron is linear.
			Knowledge of the morphology gives correct values
		3) fill neuron.bc (boundary conditions)
		'''
		branch = morphology.branch()
		i=branch._origin
		j= i + len(branch) - 2
		endpoint = j + 1
		self.neuron.index[i:endpoint+1] = self.neuron.BPcount
		children_number = 0
		
		for x in (morphology.children):#parent of segment n isnt always n-1 at branch points. We need to change Aplus and Aminus
			#gc = 2 * msiemens/cm**2
			startpoint = x._origin
			mid_diameter[startpoint] = .5*(self.neuron.diameter[endpoint]+self.neuron.diameter[startpoint])
			self.Aminus[startpoint]=mid_diameter[startpoint]**2/(4*self.neuron.diameter[startpoint]*self.neuron.length[startpoint]**2*self.neuron.Ri)
			if endpoint>0:
				self.Aplus[startpoint]=mid_diameter[startpoint]**2/(4*self.neuron.diameter[endpoint]*self.neuron.length[endpoint]**2*self.neuron.Ri)
			#else :
				#self.Aplus[startpoint]=gc
				#self.Aminus[startpoint]=gc
				
			
			children_number+=1
		
		
		pointType = self.neuron.bc[endpoint]
		hasChild = (children_number>0)
		if (not hasChild) and (pointType == 1):
			self.neuron.bc[endpoint] = self.neuron.bc_type	
		
		pointType = self.neuron.bc[endpoint]
		index_ante = self.neuron.index[ante]
		self.neuron.branches.append((i,j,endpoint,ante,index_ante,pointType))
		self.neuron.BPcount += 1

		if (j-i+2)>1:	#j-i+2 = len(branch)
			self.neuron.long_branches_count +=1
			#initialize ab
			self.ab[0,i:j]= self.Aplus[i+1:j+1]
			self.ab0[i:j] = self.Aplus[i+1:j+1]
			self.ab[2,i+1:j+1]= self.Aminus[i+1:j+1]
			self.ab2[i+1:j+1] = self.Aminus[i+1:j+1]
			
			#initialize bL
			VL0 = 1 * volt
			self.bL[i] = (- VL0 * self.Aminus[i])
			
			#initialize bR
			VR0 = 1 * volt
			self.bR[j] = (- VR0 * self.Aplus[j+1])
	
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
		self.neuron.branches = [] # (i,j,bp,ante,ante_index,pointType)
		# i is the first compartment
		# bp is the last, a branch point
		# j is the end of the "inner branch". j = bp-1
		# ante is the branch point to which i is connected
		
		self.neuron.BPcount = 0 # number of branch points (or branches). = len(self.neuron.branches)
		self.neuron.long_branches_count = 0 # number of branches with len(branch) > 1
		
		#self.vL = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		#self.vR = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		#self.d = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		
		self.bL = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		self.bR = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		#self.bd = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		self.ab = zeros((3,len(self.neuron)))
		self.ab0 = zeros(len(self.neuron))
		self.ab1 = cuda.pagelocked_zeros((len(self.neuron)),numpy.float64)
		self.ab2 = zeros(len(self.neuron))
		self.ab1_base = zeros(len(self.neuron))
		#self.res = cuda.pagelocked_zeros((3 * len(self.neuron)),numpy.float64)
		
		self.mTrunc = 0 # used to truncate vL and vR
		self.delta_list = zeros(len(self.neuron)) #used to find mTrunc
		
		# prepare_branch : fill neuron.index, neuron.branches, changes Aplus & Aminus
		self.prepare_branch(self.neuron.morphology, mid_diameter,0)
		
		# linear system P V = B used to deal with the voltage at branch points and take boundary conditions into account.
		self.P = zeros((self.neuron.BPcount,self.neuron.BPcount))
		self.B = zeros(self.neuron.BPcount)
		self.solution_bp = zeros(self.neuron.BPcount)
		
		self.gtot = zeros(len(self.neuron))
		self.I0 = zeros(len(self.neuron))
		self.i_list = []
		self.j_list = []
		self.i_list_bis = []
		self.j_list_bis = []
		new_tridiag = True
		self.bp_list = []
		self.pointType_list = []
		self.pointTypeAnte_list = []
		self.index_ante_list0 = []
		self.index_ante_list1 = []
		self.index_ante_list2 = []
		self.ante_list = []
		self.post_list = []
		self.ante_list_idx = []
		self.post_list_idx = []
		self.id = []
		self.test_list = []
		temp = zeros(self.neuron.BPcount)
		self.ind0 = []
		self.ind_bctype_0 = []
		for index,(i,j,bp,ante,index_ante,pointType) in enumerate(self.neuron.branches) :
			self.i_list.append(i)
			self.j_list.append(j)
			if new_tridiag:
				self.i_list_bis.append(i)
				ii = i
			else:
				ii = self.i_list[-1]
			if j-ii+1>2:
				self.j_list_bis.append(j)
				new_tridiag = True
			else :
				new_tridiag = False
			self.bp_list.append(bp)
			self.pointType_list.append(max(1,pointType))
			self.pointTypeAnte_list.append(max(1,self.neuron.bc[ante]))
			temp[index] = index_ante
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
		
		self.ante_arr = numpy.array(self.ante_list_idx)
		self.post_arr = numpy.array(self.post_list_idx)
		
		self.index_ante_list1, self.ind1 = numpy.unique(temp,return_index=True)
		self.ind1 = numpy.sort(self.ind1)
		self.index_ante_list1 = temp[self.ind1]
		self.index_ante_list1 = list(self.index_ante_list1)
		self.ind2 = []
		for x in xrange(self.neuron.BPcount):
			self.ind2.append(x)
		self.ind2 = numpy.delete(self.ind2,self.ind1,None) 
		self.ind2 = numpy.setdiff1d(self.ind2, self.ind0, assume_unique=True)
		self.index_ante_list2 = temp[self.ind2]
		self.index_ante_list2 = list(self.index_ante_list2)
		
		self.index_ante_list = list(temp)
		self.Aminus_bp = self.Aminus[self.bp_list]
		self.Aminus_bp [:] *= self.pointType_list[:]
		self.Aplus_i = self.Aplus[self.i_list]
		self.Aplus_i[:] *= self.pointTypeAnte_list[:]
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
		self.b_gpu_ptr = int(self.b_gpu) # bd + bL + bR -> vd + vL + vR
		
		cuda.memcpy_htod(self.ab0_gpu, ab0)
		cuda.memcpy_htod(self.ab2_gpu, ab2)
		cuda.memcpy_htod(self.bL_gpu, self.bL)
		cuda.memcpy_htod(self.bR_gpu, self.bR)
		
		self.ante_gpu = cuda.mem_alloc(self.ante_arr.size * self.ante_arr.dtype.itemsize)
		self.post_gpu = cuda.mem_alloc(self.ante_arr.size * self.ante_arr.dtype.itemsize)
		self.v_old_gpu = cuda.mem_alloc(self.neuron.v.size * dtype.itemsize)
		
		cuda.memcpy_htod(self.ante_gpu,self.ante_arr)
		cuda.memcpy_htod(self.post_gpu,self.post_arr)
		
		
		#----------------------------------------------------------------------------------
		
		self.v_branchpoints = zeros(self.neuron.BPcount)
		self.v_bp_gpu = cuda.mem_alloc(self.v_branchpoints.size * dtype.itemsize)
		
		self.timeDevice = [0]
		self.timeDeviceU = [0]
		self.timeDeviceT = [0]
		self.timeHost = [0]
		self.timeUpdater = [0]
		self.timeSolveHost = [0]
		self.timeFillHost = [0]
		self.timeFin = [0]
		
	def step(self, neuron):
		
		startDevice = time()
		start = time()
		
		if self.first_test_gtot:
			self.first_test_gtot=False
			neuron._gtot = ones(len(neuron)) * neuron._gtot
		if self.first_step :
			self.gtot = neuron._gtot
			cuda.memcpy_htod(self.gtot_gpu,self.gtot)
			self.update_ab_base_gpu()
			self.update_ab_gtot_gpu()
			
		if self.neuron.changed :
			self.I0 = neuron._I0
			cuda.memcpy_htod(self.I0_gpu,self.I0)
		
		self.update_bd_gpu()
		
		end = time()
		self.timeDeviceU.append(self.timeDeviceU[-1] + end - start)
		
		start = time()
		
		if self.first_step :
			self.calculate_vd_vL_vR_gpu()
		else:
			self.calculate_vd_gpu()
		
		end = time()
		self.timeDeviceT.append(self.timeDeviceT[-1] + end - start)
		endDevice = time()
		self.timeDevice.append(self.timeDevice[-1] + endDevice-startDevice)

		startHost = time()
		
		#-----------------------------------------------------fill P and B----------------------------------
		# 1) initPB reset P and B. every value is 0.
		# 2) fillPB calculates the coefficients of P and B where there is no lock issue.
		# 3) fillPB_bis does the same on two exclusive lists of indexes in P and B, ind1 and ind2 to avoid lock issues
		# 3) badFillPB_0 ajusts the value in the particular case of 0. this one is not parallel but not big.
		start = time()
		
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
		
		if len(self.ind_bctype_0)>0:
			self.resetPB_type0.prepared_call((len(self.ind_bctype_0), neuron.BPcount), self.P_gpu, self.B_gpu, self.v_gpu, self.ind_bctype_0_gpu,
										numpy.int32(neuron.BPcount))
			
		end = time()
		self.timeFillHost.append(self.timeFillHost[-1] + end - start)
		
		#------------------------------------------------------solve PV=B-----------------------------------
		
		cuda.memcpy_dtoh(self.P,self.P_gpu)
		cuda.memcpy_dtoh(self.B,self.B_gpu)
		
		start = time()
		self.solution_bp = solve(self.P,self.B) # better : do it on the GPU, use a sparse solver.
		end = time()
		self.timeSolveHost.append(self.timeSolveHost[-1] + end - start)
		
		cuda.memcpy_htod(self.v_bp_gpu, self.solution_bp)
		
		#-------------------------------------------------------update v-------------------------------------
		start = time()
		self.finalize_v_gpu()
		end = time()
		self.timeFin.append(self.timeFin[-1] + end - start)
		
		cuda.memcpy_dtoh(self.neuron.v,self.v_gpu)
		
		endHost = time()
		self.timeHost.append(self.timeHost[-1] + endHost - startHost)
		
		self.first_step = False
	
	def solve_branchpoints(self):
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
			
    		// fill P and B
			
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
		
	
	
	def calc(self,var,res_gpu): #compute the value of neuron.var and store it in res_gpu
		self._state_updater.calc(var,res_gpu)
		
	def update_ab_base_gpu(self):
		self.ab1_base[:-1] = (- self.neuron.Cm[:-1] / self.neuron.clock.dt * second - self.Aminus[:-1] - self.Aplus[1:])
		self.ab1_base[-1] = (- self.neuron.Cm[-1] / self.neuron.clock.dt * second - self.Aminus[-1])
		cuda.memcpy_htod(self.ab1_base_gpu,self.ab1_base)
		
	def update_ab_gtot_gpu(self):
		n = len(self.neuron)
		self.updateAB_gtot.prepared_call((n,1), self.ab1_gpu, self.ab1_base_gpu, self.gtot_gpu)
		
	def update_bd_gpu(self):
		n = len(self.neuron)
		dt = self.neuron.clock.dt
		
		self.updateBD.prepared_call((n,1), self.b_gpu, self.Cm_gpu, dt, self.v_gpu, self.I0_gpu)
		
	def calculate_vd_vL_vR_gpu(self):
		
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
	
	def calculate_vd_gpu(self):
		
		m = len(self.neuron)
		
		ab0_gpu_ptr = self.ab0_gpu_ptr
		ab1_gpu_ptr = self.ab1_gpu_ptr
		ab2_gpu_ptr = self.ab2_gpu_ptr
		b_gpu_ptr = self.b_gpu_ptr
		
		code ="""
            cusparseHandle_t handle = 0;
            cusparseCreate(&handle);
            
            double *b_gpu = (double *)b_gpu_ptr;
            double *ab0_gpu = (double *)ab0_gpu_ptr;
            double *ab1_gpu = (double *)ab1_gpu_ptr;
            double *ab2_gpu = (double *)ab2_gpu_ptr;
            
            //bd is up to date in b_gpu
            
            cusparseDgtsvStridedBatch(handle, m, ab2_gpu, ab1_gpu, ab0_gpu, b_gpu, 1, m);
            //now b_gpu contains vd, vL, vR
            cudaDeviceSynchronize();
            
            cusparseDestroy(handle);
        """
		support_code='extern "C" cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle, int m,const double *dl, const double *d,const double *du, double *x,int batchCount, int batchStride);'
		includepath = ["/usr/local/cuda/include/"]
		libpath = ['/usr/local/cuda/lib/','/usr/lib/']
		
		
		weave.inline(code,['ab0_gpu_ptr','ab1_gpu_ptr','ab2_gpu_ptr','m','b_gpu_ptr'],support_code=support_code,headers = ["<cusparse.h>","<cuda_runtime_api.h>","<cuda.h>"],
					include_dirs=includepath, library_dirs=libpath, libraries=["cusparse"], runtime_library_dirs=libpath, compiler="gcc")
			
	def calculate_vd_vL_vR_gpu_branches(self):
		
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
            	cusparseDgtsvStridedBatch(handle, n , ab2_gpu+i, ab1_gpu+i, ab0_gpu+i, b_gpu+i, 3, m);
            }
            
            //cusparseDgtsvStridedBatch(handle, m, ab2_gpu, ab1_gpu, ab0_gpu, b_gpu, 3, m);
            cudaDeviceSynchronize();
            
            cusparseDestroy(handle);
        """
		support_code='extern "C" cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle, int m,const double *dl, const double *d,const double *du, double *x,int batchCount, int batchStride);'
		includepath = ["/usr/local/cuda/include/"]
		libpath = ['/usr/local/cuda/lib/','/usr/lib/']
		
		
		weave.inline(code,['i_list','j_list','ab0_gpu_ptr','ab1_gpu_ptr','ab2_gpu_ptr','bL_gpu_ptr','bR_gpu_ptr','systemsCount','m','b_gpu_ptr'],support_code=support_code,headers = ["<cusparse.h>","<cuda_runtime_api.h>","<cuda.h>"],
					include_dirs=includepath, library_dirs=libpath, libraries=["cusparse"], runtime_library_dirs=libpath, compiler="gcc")
		
	
	def finalize_v_gpu(self):
		n = len(self.neuron)
		
		self.finalizeFun.prepared_call((n,1), self.v_gpu, self.v_bp_gpu, self.ante_gpu, self.post_gpu, self.b_gpu, numpy.int32(len(self.neuron)))
		self.finalizeFunBis.prepared_call((self.neuron.BPcount,1), self.v_gpu, self.v_bp_gpu, self.GPU_data_int)
		
		
	def had_variation(self):
		return self._state_updater.had_variation(self.neuron.clock.dt);
	
	
	def __call__(self, neuron):
		'''
		Updates the state variables.
		'''
		dt_max = 0.1 * ms
		dt_min = 0.01 * ms
		
		if not self._isprepared:
			self.prepare()
			self._isprepared=True
			print "state updater prepared"
		self.callcount+=1
		print self.callcount
		#Update I,V
		startUpdater = time()
		if neuron.changed:
			self._state_updater.changed = True
		self._state_updater(neuron)
		endUpdater = time()
		self.timeUpdater.append(self.timeUpdater[-1] + endUpdater-startUpdater)
		self.step(neuron)
		total_time = self.timeUpdater[-1]+self.timeDevice[-1]+self.timeHost[-1]
		print " tot:",total_time," U:",self.timeUpdater[-1]," D:",self.timeDevice[-1]," DU:",self.timeDeviceU[-1]," DT:",self.timeDeviceT[-1]-self.timeDeviceT[1]," H:",self.timeHost[-1]," SH:",self.timeSolveHost[-1]," FH:",self.timeFillHost[-1]," Fin:",self.timeFin[-1]
	
	def __len__(self):
		'''
		Number of state variables
		'''
		return len(self.eqs)
