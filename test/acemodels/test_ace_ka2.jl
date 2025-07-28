
using LinearAlgebra, Lux, Random, EquivariantTensors, Test, Zygote
using ACEbase.Testing: print_tf, println_slim

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))
dev = gpu_device() 

##
# generate a model 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

# generate the embedding layer 
rbasis = ET.TransformedBasis( WrappedFunction(𝐫 -> 1 / (1+norm(𝐫))), 
                              P4ML.ChebBasis(Dtot+1) )
ybasis = ET.TransformedBasis( WrappedFunction(𝐫 -> 𝐫 / norm(𝐫)), 
                              P4ML.real_solidharmonics(maxl; T = Float32, static=true) )
embed = ET.ParallelEmbed(; Rnl = rbasis, Ylm = ybasis)

mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = mb_spec, 
            Rnl_spec = P4ML.natural_indices(rbasis.basis), 
            Ylm_spec = P4ML.natural_indices(ybasis.basis), 
            basis = real )

acel = ET.SparseACElayer(𝔹basis, (1,))

model = Lux.Chain(; embed = embed, ace = acel )
ps, st = LuxCore.setup(MersenneTwister(1234), model)
θ_0 = ps.ace.WLL[1]

##
# test evaluation 

# 1. generate a random input graph 
nnodes = 100
X = ET.Testing.rand_graph(nnodes; nneigrg = 10:20)

@info("Basic ETGraph tests")
print_tf(@test ET.nnodes(X) == nnodes)
print_tf(@test ET.maxneigs(X) <= 20)
print_tf(@test ET.nedges(X) == length(X.ii) == length(X.jj) == X.first[end] - 1)
print_tf(@test all( all(X.ii[X.first[i]:X.first[i+1]-1] .== i)
                    for i in 1:nnodes ) )

##
# 2. Move model and input to the GPU / Device 
ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)

φ_dev, _ = model(X_dev, ps_dev, st_dev) 
φ_dev1 = Array(φ_dev[1])
φ, _ = model(X, ps, st) 
φ1 = φ[1]


## 
# now we try to make the same prediction with the original CPU ace 
# implementation, also skipping the graph datastructure entirely. 

function evaluate_env(model, 𝐑i)
   xij = [ rbasis.transin(𝐫, NamedTuple(), NamedTuple())[1] for 𝐫 in 𝐑i ]
   Rnl = P4ML.evaluate(rbasis.basis, xij, NamedTuple(), NamedTuple())
   𝐫̂ij = [ ybasis.transin(𝐫, NamedTuple(), NamedTuple())[1] for 𝐫 in 𝐑i ]
   Ylm = P4ML.evaluate(ybasis.basis, 𝐫̂ij, NamedTuple(), NamedTuple())
   𝔹, = ET.evaluate(𝔹basis, Rnl, Ylm) 
   return dot(𝔹, θ_0)
end

@info("Test Old Sequential vs KA Evaluation")
φ_seq = [ evaluate_env(model, ET.neighbourhood(X, i)[2]) for i in 1:nnodes ]
println_slim(@test φ1 ≈ φ_seq ≈ φ_dev1) 

##

# This passes in interactive mode but fails in a CI/test run
# φ, ∂X = ACEKA.evaluate_with_grad(model, X_dev, ps_dev, st_dev)

