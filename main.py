import numpy as np
import time 
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Define M, K, and N
M = 1024
K = 1024  
N = 1024  

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

FLOP = 2*K*M*N

start = time.perf_counter()
C = A @ B
end = time.perf_counter()
exec_time = end - start
FLOPS = FLOP/exec_time
GFLOPS = FLOPS/1e9

print(f"Execution time: {exec_time:.6f} seconds")
print(f"Performance: {GFLOPS:.2f} GFLOPS")