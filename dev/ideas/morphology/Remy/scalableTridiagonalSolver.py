import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def solve_tridiag(ld,d,ud,b,x,k,n):
    
    #will be destroyed in process : d, b.
    
    
    mod = SourceModule("""
        __global__ void pThomas(double *ld, double *d, double *ud, double *b, double *x, int gap, int n_eq)
        {
            /**
             * n_eq - number of equations
             * start - first equation
             * gap - gap between two equations
             * ld - sub-diagonal (means it is the diagonal below the main diagonal) -- indexed from 1..n-1
             * d - the main diagonal
             * ud - sup-diagonal (means it is the diagonal above the main diagonal) -- indexed from 0..n-2
             * b - right part
             * x - the answer
             */
             
            int start = threadIdx.x * gap;
            
            int nn = start + n_eq * gap;
            
            for (int i = start + gap; i < nn; i += gap)
            {
                i = start + j * gap;
                double m = ld[i]/d[i-gap];
                d[i] = d[i] - m * ud[i - gap];
                b[i] = b[i] - m*b[i-gap];
            }
            
            x[nn-gap] = b[nn-gap]/d[nn-gap];
             
            for (int i = start + (n_eq - 2)*gap; i >= start; i -= gap)
                    x[i] = (b[i] - ud[i] * x[i+gap]) / d[i];
        }
        
        __global__ void tiledPCR(double *ld, double *d, double *ud, double *b, double *x, int k, int totalSize)
        {
            int tile = pow(2,k);
            
            __shared__ double window_ld[1024]; // size : (toute la shared = 64k) / ((taille d'un element = 8) * (3 diag + un second membre))
            __shared__ double window_d[1024]; // size doit etre multiple de 4; division par 2 pour ?
            __shared__ double window_ud[1024]; 
            __shared__ double window_b[1024];
            
            double k1;
            double k2;
            int gap;
            int ofst_read;
            int pow_i;
            int ofst_write;
            int offset = 0;
            int idx = threadIdx.x; //threadIdx.x va de 0 a 255
            
            //initialyze window
            window_ld[idx-128] = ld[idx];
            window_ld[idx+128] = ld[idx+256];
            window_d[idx-128] = d[idx];
            window_d[idx+128] = d[idx+256];
            window_ud[idx-128] = ud[idx];
            window_ud[idx+128] = ud[idx+256];
            window_b[idx-128] = b[idx];
            window_b[idx+128] = b[idx+256];
            
            int pow_i_0 = 2;
            int ofst_write_0 = 0;
            for(int i=2;i<(k+1);i++)
            {
                pow_i_0 *= 2;
                ofst_write_0 += pow_i_0;
            }
            
            //---------------------------cas offset=0-----------------------------
            
            
            //---------------------------FIN cas offset=0-----------------------------
            //----------------------------cas general-----------------------------
            while(offset < totalSize - tile) // ?
            {
                gap = 1;
                pow_i = pow_i_0;
                ofst_write = ofst_write_0;
                ofst_read = 0;
                for(int iter=0;iter<(k-1);iter++) //k=8; 256 threads
                {
                    k1 = window_ld[ofst_read+idx]/window_d[ofst_read+idx-gap];
                    k2 = window_ud[ofst_read+idx]/window_d[ofst_read+idx+gap];
                    window_ld[ofst_write+idx] = - window_ld[ofst_read+idx-gap]*k1;
                    window_d[ofst_write+idx] = window_d[ofst_read+idx] - window_ud[ofst_read+idx-gap]*k1 - window_ld[ofst_read+idx+gap] * k2;
                    window_ud[ofst_write+idx] = - window_ud[ofst_read+idx+gap]*k2;
                    window_b[ofst_write+idx] = window_b[ofst_read+idx] - window_b[ofst_read+idx-gap]*k1 - window_b[ofst_read+idx+gap] * k2;
                    
                    ofst_write = ofst_write - pow_i;
                    pow_i /= 2;
                    ofst_read = ofst_write + pow_i;
                    gap *= 2;
                    
                    __syncthreads();
                }
                //iter = k-1 :
                gap *= 2;
                k1 = window_ld[idx]/window_d[idx-gap];
                k2 = window_ud[idx]/window_d[idx+gap];
                ld[idx-256+offset] = - window_ld[idx-gap]*k1;
                d[idx-256] = window_d[idx] - window_ud[idx-gap]*k1 - window_ld[idx+gap] * k2;
                ud[idx-256] = - window_ud[idx+gap]*k2;
                b[idx-256] = window_b[idx] - window_b[idx-gap]*k1 - window_b[idx+gap] * k2;
                
                __syncthreads();
                //on fait quoi avec window ? on stocke les systemes resultats ou ?
                
                
                //decaler window. ne pas le faire dans tous les threads !
                for(int i = 0; i<768; i++)
                {
                    window_ld[i] = window_ld[i+256];
                    window_d[i] = window_d[i+256];
                    window_ud[i] = window_ud[i+256];
                    window_b[i] = window_b[i+256];
                }
                for(int i = 768; i<1024; i++)
                {
                    window_ld[i] = ld[i+offset]; //offset ?
                }
                
                offset += 256;
            }
            //----------------------------FIN cas general-----------------------------
            //----------------------------cas offset = max----------------------------
            //---------------------FIN cas offset = max-----------------------
        }
    """)
    pThomas = mod.get_function("pThomas")
    pThomas.prepare(['P','P','P','P','P','i','i'],block=(2**k,1,1))
    
    pThomas.prepared_call(ld,d,ud,b,x,2**k,n/(2**k))
    
    