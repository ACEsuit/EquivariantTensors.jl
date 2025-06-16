
using SparseArrays, LinearAlgebra, Test 
using ACEbase.Testing: print_tf
import EquivariantTensors as ET

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

## 

@info("Testing DevSparseMatrixCSR multiplication")

for itest = 1:20 
   local A, B 
   m = rand(50:200)
   n = rand(100:300)
   nB = rand(30:100)
   p = 0.01 * (1 + rand())

   A = sprand(m, n, p) 
   A_dev = ET.DevSparseMatrixCSR(A, dev)
   B = randn(Float32, n, nB) 
   B_dev = dev(B)

   X = A * B
   X_dev = ET.mul(A_dev, B_dev)
   print_tf(@test X ≈ Array(X_dev))
end
println() 
