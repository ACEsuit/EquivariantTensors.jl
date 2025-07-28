# TODO: 
#   - move the ACEKA into the main source code 
#   - use named tuples
#   - gradients 
# actually, this is likely obsolete due to test_ace_ka.jl so check this and 
# then remove this file. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random, Test

# using Zygote, LuxCore, Lux
# import Optimisers as OPT
# import ForwardDiff as FDiff 

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))
dev = gpu_device() 

@info("TEST Implementation of Sparse ACE with Graph input and KA eval")

##

#
# temporarily here in the test file; to be moved into main source 
# to make the tests look clean and neat and nice. Basically, SimpleACE 
# needs to be replaced with a SparseACElayer. 
#
#
module ACEKA

   using LinearAlgebra, Random 
   import LuxCore: initialparameters, initialstates

   import EquivariantTensors as ET 

   struct SimpleACE{T, TEM, BB}
      embed::TEM          # assume Rnl Ylm embedding 
      symbasis::BB        # assume SparseACEbasis(L = 0)
      params::Vector{T}   # model parameters
   end

   initialparameters(rng::AbstractRNG, m::SimpleACE) = 
            (    embed = initialparameters(rng, m.embed), 
              symbasis = initialparameters(rng, m.symbasis), 
                params = copy(m.params), )

   initialstates(rng::AbstractRNG, m::SimpleACE) = 
            (    embed = initialparameters(rng, m.embed), 
              symbasis = initialstates(rng, m.symbasis), )

   # ---------------------------------
   # evaluation code

   function evaluate(model::SimpleACE, X::ET.ETGraph, ps, st)
      (Rn_3, Ylm_3), _ = ET.evaluate(model.embed, X, ps.embed, st.embed)
      𝔹, _ = ET.ka_evaluate(model.symbasis, Rn_3, Ylm_3, ps.symbasis, st.symbasis)
      return 𝔹 * ps.params, st 
   end

end


## 
Dtot = 16
maxl = 10
ORD = 3 

rbasis = P4ML.ChebBasis(Dtot+1)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)
embed = ET.RnlYlmEmbedding(𝐫 -> 1 / (1+norm(𝐫)), rbasis, 
                           𝐫 -> 𝐫 / norm(𝐫), ybasis)

# generate the nnll basis pre-specification
mb_spec = ET.sparse_nnll_set(; ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = 0, mb_spec = mb_spec, 
            Rnl_spec = P4ML.natural_indices(rbasis), 
            Ylm_spec = P4ML.natural_indices(ybasis), 
            basis = real )

θ = randn(Float32, length(𝔹basis, 0))

model = ACEKA.SimpleACE(embed, 𝔹basis, θ)
ps, st = LuxCore.setup(MersenneTwister(1234), model)

##
# test evaluation 

# 1. generate a random input graph 
X = ET.Testing.rand_graph(100)

# 2. Move model and input to the GPU / Device 
ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)

# 3. run forwardpass through the model
φ, _ = ACEKA.evaluate(model, X, ps, st) 
φ_dev, _ = ACEKA.evaluate(model, X_dev, ps_dev, st_dev) 

println_slim(@test φ ≈ Array(φ_dev))
