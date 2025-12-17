
using LinearAlgebra, Lux, Random, EquivariantTensors, Test, StaticArrays,
      Zygote, ForwardDiff, DecoratedParticles
using ACEbase.Testing: print_tf, println_slim

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

##

module ACEKA

   using LinearAlgebra, Random, Zygote  
   using DecoratedParticles: VState
   import LuxCore: initialparameters, initialstates
   import ChainRulesCore: rrule

   import EquivariantTensors as ET 
   import KernelAbstractions as KA

   struct SimpleACE{T, TR, TY, BB}
      Rnl::TR 
      Ylm::TY
      symbasis::BB    # symmetric basis 
      params::Vector{T}   # model parameters
   end

   initialparameters(rng::AbstractRNG, m::SimpleACE) = 
            (      Rnl = initialparameters(rng, m.Rnl), 
                   Ylm = initialparameters(rng, m.Ylm), 
              symbasis = initialparameters(rng, m.symbasis), 
                params = copy(m.params), )

   initialstates(rng::AbstractRNG, m::SimpleACE) = 
            (      Rnl = initialstates(rng, m.Rnl), 
                   Ylm = initialstates(rng, m.Ylm), 
              symbasis = initialstates(rng, m.symbasis), )

   # 𝔹 = (#nodes, #features); params = (#features, #readouts)
   # in this toy model, #readouts = 1.

   function eval_basis(model::SimpleACE, X::ET.ETGraph, ps, st) 
      Rnl, _ = model.Rnl(X, ps.Rnl, st.Rnl)
      Ylm, _ = model.Ylm(X, ps.Ylm, st.Ylm)
      (𝔹,), _ = ET.ka_evaluate(model.symbasis, Rnl, Ylm, ps.symbasis, st.symbasis)
      return 𝔹
   end               

   function evaluate(model::SimpleACE, X::ET.ETGraph, ps, st)
      𝔹 = eval_basis(model, X, ps, st)
      return 𝔹 * ps.params, st 
   end

   function jacobian_basis(model::SimpleACE, X::ET.ETGraph, ps, st) 
      (R, _∂R), _ = ET.evaluate_ed(model.Rnl, X, ps.Rnl, st.Rnl)
      (Y, _∂Y), _ = ET.evaluate_ed(model.Ylm, X, ps.Ylm, st.Ylm)
      ∂R = VState.( _∂R )
      ∂Y = VState.( _∂Y )
      𝔹, ∂𝔹 = ET._jacobian_X(model.symbasis, R, Y, ∂R, ∂Y)
      return 𝔹, ∂𝔹
   end


   # Semi-manual gradients are still much more efficient 
   #
   function evaluate_with_grad(model::SimpleACE, X::ET.ETGraph, ps, st)
      (Rnl, dRnl), _ = ET.evaluate_ed(model.Rnl, X, ps.Rnl, st.Rnl)
      (Ylm, dYlm), _ = ET.evaluate_ed(model.Ylm, X, ps.Ylm, st.Ylm)

      (𝔹,), A, 𝔸 = ET._ka_evaluate(model.symbasis, Rnl, Ylm, 
                  st.symbasis.aspec, st.symbasis.aaspecs, st.symbasis.A2Bmaps)
      φ = 𝔹 * ps.params

      # let's assume we eventually produce E = ∑φ then ∂E = 1, which 
      # backpropagates to ∂φ = (1,1,1...)
      # ∂E/∂𝔹 = ∂/∂𝔹 { 1ᵀ 𝔹 params } = ∂/∂𝔹 { 𝔹 : 1 ⊗ params}
      ∂𝔹 = fill!(similar(𝔹, (size(𝔹, 1),)), one(eltype(𝔹))) * ps.params' 

      # packpropagate through the symmetric basis 
      ∂Rnl, ∂Ylm = ET._ka_pullback((∂𝔹,), model.symbasis, Rnl, Ylm, A, 𝔸, 
                                    st.symbasis.aspec, st.symbasis.aaspecs, st.symbasis.A2Bmaps)

      # this could be made more memory efficient by avoiding the 
      # many intermediate allocations 
      _grad_R = ET._pullback_edge_embedding(∂Rnl, VState.(dRnl), X) 
      _grad_Y = ET._pullback_edge_embedding(∂Ylm, VState.(dYlm), X)

      return φ, _grad_R .+ _grad_Y
   end       

end


##
# generate a model 
Dtot = 12   # total degree; specifies the trunction of embeddings and correlations
maxl = 8    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

# generate the embedding layer 

rbasis = P4ML.ChebBasis(Dtot+1)
rtrans = ET.NTtransform(x -> 1 / (1+norm(x.𝐫)^2))
rembed = ET.EdgeEmbed( ET.EmbedDP(rtrans, rbasis); name = "Rnl" )

ybasis = P4ML.real_solidharmonics(maxl; static=true)
ytrans = ET.NTtransform(x -> x.𝐫)
yembed = ET.EdgeEmbed( ET.EmbedDP(ytrans, ybasis); name = "Ylm" )

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

model = ACEKA.SimpleACE(rembed, yembed, 𝔹basis, θ)
_ps, _st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(_ps); st = ET.float32(_st)

##
# test evaluation 

# 1. generate a random input graph 
nnodes = 30
_X = ET.Testing.rand_graph(nnodes; nneigrg = 5:10)
X = ET.float32(_X)

@info("Basic ETGraph tests")
print_tf(@test ET.nnodes(X) == nnodes)
print_tf(@test ET.maxneigs(X) <= 20)
print_tf(@test ET.nedges(X) == length(X.ii) == length(X.jj) == X.first[end] - 1)
print_tf(@test all( all(X.ii[X.first[i]:X.first[i+1]-1] .== i)
                    for i in 1:nnodes ) )
println()                      

##
# 2. Move model and input to the GPU / Device 
ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)

@info("Test KA Model Evaluation on CPU and Device")
φ, _ = ACEKA.evaluate(model, X, ps, st)
φ_dev, _ = ACEKA.evaluate(model, X_dev, ps_dev, st_dev) 
φ_dev1 = Array(φ_dev)

println_slim(@test φ ≈ φ_dev1)

## 
# now we try to make the same prediction with the original CPU ace 
# implementation, also skipping the graph datastructure entirely. 

function _basis_env(model::ACEKA.SimpleACE, 𝐑i)
   rij = [ rtrans(x) for x in 𝐑i ]
   Rnl = P4ML.evaluate(rbasis, rij)
   𝐫ij = [ x.𝐫 for x in 𝐑i ]
   Ylm = P4ML.evaluate(ybasis, 𝐫ij)
   𝔹, = ET.evaluate(𝔹basis, Rnl, Ylm)   
   return 𝔹
end

function evaluate_env(model::ACEKA.SimpleACE, 𝐑i)
   𝔹 = _basis_env(model, 𝐑i)
   return dot(𝔹, θ)
end


@info("Test Old Sequential vs KA Evaluation")
φ_seq = [ evaluate_env(model, ET.neighbourhood(X, i)[2]) for i in 1:nnodes ]
println_slim(@test φ ≈ φ_seq ≈ φ_dev1) 

##

# semi-hand-written gradient 
# this currently doesn't run on GPU yet -> urgent TODO 
φ, ∂X = ACEKA.evaluate_with_grad(model, X, ps, st)


##

@info("Test Differentiation through KA Model Evaluation")

@info("Zygote.gradient") 
energy(model, G, ps, st) = sum(ACEKA.evaluate(model, G, ps, st)[1])
∇E_zy = Zygote.gradient(G -> energy(model, G, ps, st), X)[1] 

# This requires a fix to P4ML to work properly on the GPU device
# ∇E_zy_dev = Zygote.gradient(G -> energy(model, G, ps_dev, st_dev), X_dev)[1] 


##

@info("ForwardDiff") 

function replace_edges(X, Rmat)
   Rsvec = [ SVector{3}(Rmat[:, i]) for i in 1:size(Rmat, 2) ]
   new_edgedata = [ (; 𝐫 = 𝐫) for 𝐫 in Rsvec ]
   return ET.ETGraph( X.ii, X.jj, X.first, 
               X.node_data, new_edgedata, X.graph_data, 
               X.maxneigs )
end 

function grad_fd(model, G) 
   function _energy(Rmat)
      G_new = replace_edges(G, Rmat)
      return sum(ACEKA.evaluate(model, G_new, ps, st)[1])
   end
      
   Rsvec = [ x.𝐫 for x in G.edge_data ]
   Rmat = reinterpret(reshape, eltype(Rsvec[1]), Rsvec)
   ∇E_fd = ForwardDiff.gradient(_energy, Rmat)
   ∇E_svec = [ SVector{3}(∇E_fd[:, i]) for i in 1:size(∇E_fd, 2) ]
   ∇E_edges = [ (; 𝐫 = 𝐫) for 𝐫 in ∇E_svec ]
   return ET.ETGraph( G.ii, G.jj, G.first, 
               G.node_data, ∇E_edges, G.graph_data, 
               G.maxneigs )
end 

∇E_fd = grad_fd(model, X)

##

@info("Confirm FD and Zygote agree")
∇E_zy_𝐫 = [ x.𝐫 for x in ∇E_zy.edge_data ] 
∇E_fd_𝐫 = [ x.𝐫 for x in ∇E_fd.edge_data ]
∇E_man_𝐫 = [ x.𝐫 for x in ∂X ]

println_slim(@test all(∇E_fd_𝐫 .≈ ∇E_zy_𝐫 .≈ ∇E_man_𝐫 ))

##

@info("Test Jacobian of basis w.r.t. positions")

# to test the jacobian, we check whether it gives the 
# same as the gradient after contraction with the parameters 
# first we transform it into edge format 

# jacobian as 3-dim tensor 
𝔹3, ∂𝔹3 = ACEKA.jacobian_basis(model, X, ps, st)

# convert to 2-dimensional tensor (compat with ∇E_zy)
∂𝔹2 = ET.rev_reshape_embedding(∂𝔹3[1], X)
∂𝔹2xθ = ∂𝔹2 * θ

println_slim(@test 𝔹3[1] ≈ ACEKA.eval_basis(model, X, ps, st))
println_slim(@test all(VState.(∇E_zy.edge_data) .≈ ∂𝔹2xθ)) 

##

# This is reasonably efficient, but would be good to reduce the allocations  
@info("Timings")
println(" Basis: ")
@time ACEKA.eval_basis(model, X, ps, st)
println(" Evaluate with Grad: ")
@time ACEKA.evaluate_with_grad(model, X, ps, st)
println(" Zygote Gradient: ")
@time Zygote.gradient(G -> energy(model, G, ps, st), X)[1] 
println(" Jacobian Basis: ")
@time ACEKA.jacobian_basis(model, X, ps, st)
