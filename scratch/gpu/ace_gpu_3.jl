# First prototype implementation of a pure GPU ACE implementation. 
#

using LinearAlgebra, StaticArrays, Metal, Lux, 
      Combinatorics, Random, EquivariantTensors

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

dev = gpu_device() 

##

module ACEKA

   using LinearAlgebra, Random 
   import LuxCore: initialparameters, initialstates

   import EquivariantTensors as ET 

   struct SimpleACE{T, TEM, BB}
      embed::TEM
      symbasis::BB    # symmetric basis 
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

   function evaluate(model::SimpleACE, X::ET.PtClGraph, ps, st)
      (Rn_3, Ylm_3), _ = ET.evaluate(model.embed, X, ps.embed, st.embed)
      𝔹, _ = ET.ka_evaluate(model.symbasis, Rn_3, Ylm_3, ps.symbasis, st.symbasis)
      return transpose(𝔹) * ps.params, st 
   end

end



##
# generate a model 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

# generate the embedding layer 
rbasis = P4ML.ChebBasis(Dtot+1)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)
embed = ET.RnlYlmEmbedding(𝐫 -> 1 / (1+norm(𝐫)), rbasis, 
                           𝐫 -> 𝐫 / norm(𝐫), ybasis)

mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = mb_spec, 
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
φ, _ = ACEKA.evaluate(model, X_dev, ps_dev, st_dev) 

