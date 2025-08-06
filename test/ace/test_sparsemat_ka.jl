
using SparseArrays, LinearAlgebra, Test 
using ACEbase.Testing: print_tf
import EquivariantTensors as ET
import KernelAbstractions as KA
import Random 

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

## 

@info("Testing DevSparseMatrixCSR multiplication")

Random.seed!(6)
for itest = 1:100 
   local A, B, X, Y   
   m = rand(50:200)
   n = rand(100:300)
   nB = rand(30:100)
   p = 0.01 * (1 + rand())

   A = sprand(Float32, m, n, p)
   A_dev = ET.DevSparseMatrixCSR(A, dev)
   B = randn(Float32, n, nB) 
   B_dev = dev(B)
   C = randn(Float32, nB, m)
   C_dev = dev(C)

   X = A * B
   X_dev = ET.mul(A_dev, B_dev)
   print_tf(@test X ≈ Array(X_dev))

   Y = C * A
   Y_dev = ET.mul(C_dev, A_dev)
   KA.synchronize(KA.get_backend(Y_dev))
   print_tf(@test Y ≈ Array(Y_dev))
end
println() 
