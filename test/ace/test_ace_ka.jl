
using LinearAlgebra, Metal, Lux, Random, EquivariantTensors, Test, Zygote
using ACEbase.Testing: print_tf, println_slim

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

dev = gpu_device() 

##

module ACEKA

   using LinearAlgebra, Random, Zygote  
   import LuxCore: initialparameters, initialstates
   import ChainRulesCore: rrule

   import EquivariantTensors as ET 
   import KernelAbstractions as KA

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

   function evaluate(model::SimpleACE, X::ET.ETGraph, ps, st)
      (Rn_3, Ylm_3), _ = ET.evaluate(model.embed, X, ps.embed, st.embed)
      𝔹, _ = ET.ka_evaluate(model.symbasis, Rn_3, Ylm_3, ps.symbasis, st.symbasis)
      # 𝔹 = (#nodes, #features); params = (#features, #readouts)
      # in this toy model, #readouts = 1.
      return 𝔹 * ps.params, st 
   end

   function evaluate_with_grad(model::SimpleACE, X::ET.ETGraph, ps, st)
      backend = KA.get_backend(ps.params)
      (Rnl_3, Ylm_3), _ = ET.evaluate(model.embed, X, ps.embed, st.embed)
      𝔹, A, 𝔸 = ET._ka_evaluate(model.symbasis, Rnl_3, Ylm_3, 
               st.symbasis.aspec, st.symbasis.aaspecs, st.symbasis.A2Bmaps[1]) 
      φ = 𝔹 * ps.params
      # let's assume we eventually produce E = ∑φ then ∂E = 1, which 
      # backpropagates to ∂φ = (1,1,1...)
      # ∂E/∂𝔹 = ∂/∂𝔹 { 1ᵀ 𝔹 params } = ∂/∂𝔹 { 𝔹 : 1 ⊗ params}
      ∂𝔹 = KA.ones(backend, eltype(𝔹), (size(𝔹, 1),)) * ps.params' 

      # packpropagate through the symmetric basis 
      (∂Rnl_3, ∂Ylm_3), _ = ET.ka_pullback(∂𝔹, model.symbasis, 
                                           Rnl_3, Ylm_3, A, 𝔸, 
                                           ps.symbasis, st.symbasis) 
      ∂X, _ = ET.ka_pullback( ∂Rnl_3, ∂Ylm_3, model.embed, 
                              X, ps.embed, st.embed)

      return φ, ∂X
   end
end


##
# generate a model 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

# generate the embedding layer 
rtrans = 𝐫 -> 1 / (1+norm(𝐫))
rbasis = P4ML.ChebBasis(Dtot+1)
ytrans = 𝐫 -> 𝐫 / norm(𝐫)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)
embed = ET.RnlYlmEmbedding(rtrans, rbasis, ytrans, ybasis)

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

# 3. run forwardpass through the model
φ_dev, _ = ACEKA.evaluate(model, X_dev, ps_dev, st_dev) 
φ = Array(φ_dev)

## 
# now we try to make the same prediction with the original CPU ace 
# implementation, also skipping the graph datastructure entirely. 

function evaluate_env(model::ACEKA.SimpleACE, 𝐑i)
   xij = [ rtrans(𝐫) for 𝐫 in 𝐑i ]
   Rnl = P4ML.evaluate(rbasis, xij)
   𝐫̂ij = [ ytrans(𝐫) for 𝐫 in 𝐑i ]
   Ylm = P4ML.evaluate(ybasis, 𝐫̂ij)
   𝔹, = ET.evaluate(𝔹basis, Rnl, Ylm) 
   return dot(𝔹, θ)
end

@info("Test Old Sequential vs KA Evaluation")
φ_seq = [ evaluate_env(model, ET.neighbourhood(X, i)[2]) for i in 1:nnodes ]
println_slim(@test φ ≈ φ_seq) 

##

φ, ∂X = ACEKA.evaluate_with_grad(model, X_dev, ps_dev, st_dev)

