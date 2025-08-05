
using LinearAlgebra, Lux, Random, EquivariantTensors, Test, Zygote, StaticArrays
using ACEbase.Testing: print_tf, println_slim

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

##
# generate a model 
Dtot = 16    # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3      # correlation-order (body-order = ORD + 1)

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
ps = ET.float32(ps); st = ET.float32(st)
θ_0 = ps.ace.WLL[1] # for testing only 

##
# test evaluation 

# 1. generate a random input graph 
nnodes = 100
X = ET.Testing.rand_graph(nnodes; nneigrg = 10:20)

@info("Basic ETGraph tests")
println_slim(@test ET.nnodes(X) == nnodes)
println_slim(@test ET.maxneigs(X) <= 20)
println_slim(@test ET.nedges(X) == length(X.ii) == length(X.jj) == X.first[end] - 1)
println_slim(@test all( all(X.ii[X.first[i]:X.first[i+1]-1] .== i)
                    for i in 1:nnodes ) )

##
# 2. Move model and input to the GPU / Device 
ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)

# evaluate on CPU and GPU 
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
   return transpose(𝔹) * θ_0
end

@info("Test Old Sequential vs KA Evaluation")
φ_seq = [ evaluate_env(model, ET.neighbourhood(X, i)[2])[1] for i in 1:nnodes ]
println_slim(@test φ1 ≈ φ_seq ≈ φ_dev1)

##
# Check gradient w.r.t. parameters 

Δ = randn(Float32, size(φ1))
Δ_dev = dev(Δ)
function _foo(_ps) 
   out1 = model(X, _ps, st)
   val = out1[1][1]
   dot(val, Δ)
end 
_foo_dev(_ps) = dot(model(X_dev, _ps, st_dev)[1][1], Δ_dev)
println_slim(@test _foo(ps) ≈ _foo_dev(ps_dev))

Zygote.gradient(_foo, ps)


##  

@info("Test multiple outputs")

acel = ET.SparseACElayer(𝔹basis, (8,))

model = Lux.Chain(; embed = embed, ace = acel )
ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)
θ_0 = ps.ace.WLL[1] # for testing only 

ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)
φ_dev, _ = model(X_dev, ps_dev, st_dev) 
φ_dev1 = Array(φ_dev[1])
φ, _ = model(X, ps, st) 
φ1 = φ[1]
φ_seq = reduce(vcat, 
         [ evaluate_env(model, ET.neighbourhood(X, i)[2]) for i in 1:nnodes ])
println_slim(@test φ1 ≈ φ_seq ≈ φ_dev1)

##  

@info("Test equivariant outputs")

# 4 scalars (L=0), 2 vectors (L=1)
# NOTE: sparse_nnll_set cannot manage a simplification by passing in 
#       the LL tuple; this should be added to make basis generation more 
#       efficient. 
LL = (0, 1)
NFEAT = (4, 2) 
mb_spec = ET.sparse_nnll_set(; ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = LL, mb_spec = mb_spec, 
            Rnl_spec = P4ML.natural_indices(rbasis.basis), 
            Ylm_spec = P4ML.natural_indices(ybasis.basis), 
            basis = real )

acel = ET.SparseACElayer(𝔹basis, NFEAT)

model = Lux.Chain(; embed = embed, ace = acel )
ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)
θ_0 = ps.ace.WLL[1] # for testing only 

ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)
φ_dev, _ = model(X_dev, ps_dev, st_dev) 
φ_dev1 = Array.(φ_dev)

φ, _ = model(X, ps, st) 
println_slim(@test size(φ[1]) == (nnodes, NFEAT[1]))
println_slim(@test size(φ[2]) == (nnodes, NFEAT[2]))
println_slim(@test eltype(φ[1]) == Float32)
println_slim(@test eltype(φ[2]) == SVector{3, Float32})
println_slim(@test all(φ_dev1 .≈ φ))
