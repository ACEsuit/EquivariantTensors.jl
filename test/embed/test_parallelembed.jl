using LinearAlgebra, Lux, Random, EquivariantTensors, Test, Zygote
using ACEbase.Testing: print_tf, println_slim

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

# include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))
# dev = gpu_device() 

##
# generate a model 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 

# generate the embedding layer 
rtrans = 𝐫 -> 1 / (1+norm(𝐫))
rbasis = P4ML.ChebBasis(Dtot+1)
ytrans = 𝐫 -> 𝐫 / norm(𝐫)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)

embed1 = ET.RnlYlmEmbedding(rtrans, rbasis, ytrans, ybasis)
ps1, st1 = LuxCore.setup(MersenneTwister(1234), embed1)

embed2 = ET.ParallelEmbed(; 
      Rnl = ET.TransformedBasis(; transin = WrappedFunction(rtrans), basis = rbasis), 
      Ylm = ET.TransformedBasis(; transin = WrappedFunction(ytrans), basis = ybasis), 
      name = "RnlYlm" )
ps2, st2 = LuxCore.setup(MersenneTwister(1234), embed2)

##

nnodes = 100
X = ET.Testing.rand_graph(nnodes; nneigrg = 10:20)

(R1, Y1), _ = ET.evaluate(embed1, X, ps1, st1)
(R2, Y2), _ = ET.evaluate(embed2, X, ps2, st2)

# sequential evaluation
R0 = fill!(similar(R1), 0) 
Y0 = fill!(similar(Y1), 0)

let R0 = R0, Y0 = Y0 
   idx = 0 
   for i = 1:nnodes 
      idxj = 0 
      for j = X.first[i]:X.first[i+1]-1
         idx += 1; idxj += 1
         x = X.edge_data[idx]
         R0[idxj, i, :] = rbasis(rtrans(x))
         Y0[idxj, i, :] = ybasis(ytrans(x))
      end
   end
end 

println_slim(@test R0 ≈ R1 ≈ R2)
println_slim(@test Y0 ≈ Y1 ≈ Y2)

