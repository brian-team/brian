'''
The code below uses the GPU to compute a matrix-matrix product.

Notes:

- The C code is adapted from the simpleCUBLAS sample in the CUDA SDK
- Matrices have to have Fortran ordering [is this equivalent to
  transposing? if so this is not a problem because BLAS includes
  the option to operate on transposed matrices]
- Timing comparisons are for the time required to initialise CUDA,
  upload, perform the operation, and then download again, so for
  small times the CPU performs better. On mine at n=1000 it's about
  the same, but for N=4000 the GPU is 10x faster.
- I'm not sure how you're supposed to indicate failure and return
  in a Weave inline function, so that's just ignored for the moment,
  if one of the functions returns an error the program will
  probably just crash.
'''

from numpy import *
from scipy import weave
from scipy import randn
import time

n=4000
x=array(randn(n, n), order='F')
y=array(randn(n, n), order='F')
z=array(zeros((n, n)), order='F')

code='''
cublasStatus status;
double *d_x = 0;
double *d_y = 0;
double *d_z = 0;
double alpha = 1.0;
double beta = 0.0;
int n2 = n*n; 

//fprintf (stderr, "fprintf working\\n");

status = cublasInit();
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! CUBLAS initialization error\\n");
    //return EXIT_FAILURE;
}

//////////////////// Alloc device mem ///////////////////////////
status = cublasAlloc(n2, sizeof(d_x[0]), (void**)&d_x);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (x)]\\n");
    //return EXIT_FAILURE;
}
status = cublasAlloc(n2, sizeof(d_y[0]), (void**)&d_y);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (y)]\\n");
    //return EXIT_FAILURE;
}
status = cublasAlloc(n2, sizeof(d_z[0]), (void**)&d_z);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (z)]\\n");
    //return EXIT_FAILURE;
}

//////////////////// Upload ///////////////////////////
status = cublasSetVector(n2, sizeof(x[0]), x, 1, d_x, 1);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device access error (write x)\\n");
    //return EXIT_FAILURE;
}
status = cublasSetVector(n2, sizeof(y[0]), y, 1, d_y, 1);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device access error (write y)\\n");
    //return EXIT_FAILURE;
}

//////////////////// Multiply ///////////////////////////
cublasDgemm('n', 'n', n, n, n, alpha, d_x, n, d_y, n, beta, d_z, n);
status = cublasGetError();
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\\n");
    //return EXIT_FAILURE;
}

//////////////////// Download ///////////////////////////
status = cublasGetVector(n2, sizeof(z[0]), d_z, 1, z, 1);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device access error (read z)\\n");
    //return EXIT_FAILURE;
}

//////////////////// Free memory ///////////////////////////
status = cublasFree(d_x);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (x)\\n");
    //return EXIT_FAILURE;
}
status = cublasFree(d_y);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (y)\\n");
    //return EXIT_FAILURE;
}
status = cublasFree(d_z);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (z)\\n");
    //return EXIT_FAILURE;
}

/* Shutdown */
status = cublasShutdown();
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error \\n");
    //return EXIT_FAILURE; 
}

'''

start=time.time()
weave.inline(code, ['x', 'y', 'z', 'n'],
             compiler='gcc', #msvc works too
             headers=['"cublas.h"'],
             include_dirs=['C:\\CUDA\\include'],
             libraries=['cublas'],
             library_dirs=['C:\\CUDA\\lib'],
             )
end=time.time()

print 'GPU time:', end-start

start=time.time()
w=dot(x, y)
end=time.time()

print 'CPU time:', end-start

print 'Max abs difference in values:', amax(abs(w-z))

print 'OK'
