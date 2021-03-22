import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import sys
from time import perf_counter
from math import ceil
from pycuda import gpuarray
from pycuda.compiler import SourceModule


MATRIX_SIZE = int(sys.argv[1])

a_mat = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_mat = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c_mat = a_mat.dot(b_mat)
#a_mat = np.array([[1,2],[3,4]], dtype=np.float32)
#b_mat = np.array([[5,6],[7,8]], dtype=np.float32)
a_mat_gpu = gpuarray.to_gpu(a_mat)
b_mat_gpu = gpuarray.to_gpu(b_mat)
output_mat_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

ker = SourceModule("""

                   __global__ void scalar_multiplication_kernel(
                   float *a_mat, float *b_mat, float *output_mat){
                    float a_val, b_val;
                    int tx = blockIdx.x*blockDim.x+threadIdx.x;
                    int ty = blockIdx.y*blockDim.y+threadIdx.y;
                    float pval = 0;
                    if (tx < %(MATRIX_SIZE)s && ty < %(MATRIX_SIZE)s){
                     for (int k=0; k < %(MATRIX_SIZE)s; ++k){
                      a_val = a_mat[tx * %(MATRIX_SIZE)s + k];
                      b_val = b_mat[k * %(MATRIX_SIZE)s + ty];
                      pval += a_val * b_val;
                     }
                     output_mat[tx * %(MATRIX_SIZE)s + ty] = pval; 
                    }
                   }"""%{'MATRIX_SIZE': MATRIX_SIZE})

scalar_multiplu_gpu = ker.get_function("scalar_multiplication_kernel")

start = drv.Event()
end = drv.Event()

#start.record()
start = perf_counter()
scalar_multiplu_gpu(a_mat_gpu, b_mat_gpu, output_mat_gpu, block=(32,32,1), grid=(ceil(MATRIX_SIZE/32),ceil(MATRIX_SIZE/32),1))
end = perf_counter()
# end.record()
# end.synchronize()
# secs = start.time_till(end)*1e-3
# output = np.allclose(c_mat,output_mat_gpu.get())

a_mat_gpu.gpudata.free()
b_mat_gpu.gpudata.free()
output_mat_gpu.gpudata.free()

print(end-start)
