# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model "by hand" (as opposed to via an ML framework)
# Here we use all the utility functions that the ET library offers, 
# whereas in `simple_ace.jl` we do most steps by hand. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random
using ChainRulesCore: rrule

##

# This struct defines a simple ACE-like model. The inputs are a cloud of points
# 𝐫 = (r₁, r₂, ..., rₙ) in 3D space. The output of the model is a scalar that 
# is invariant under rotations, reflections and permutations. 

struct SimpleACE2{T, RB, YB, BB}
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   symbasis::BB    # symmetric basis 
   params::Vector{T}   # model parameters
end


function eval_with_grad(m::SimpleACE2, 𝐫::AbstractVector{<: SVector{3}})
   # [1] Embeddings: evaluate the Rn and Ylm embeddings
   #   Rn[j] = Rn(norm(𝐫[j])), Ylm[j] = Ylm(Rs[j])
   r = norm.(𝐫)
   𝐲 = 𝐫
   Rn = P4ML.evaluate(m.rbasis, r)
   Ylm = P4ML.evaluate(m.ybasis, 𝐲)

   # [2] feed the Rn, Ylm embeddings through the sparse ACE model 
   #     but we do this via an rrule so we get the pullback for free
   𝔹, pb_𝔹 = rrule(ET.evaluate, m.symbasis, Rn, Ylm)
   
   # [3] the model output value is the dot product with the parameters 
   φ = dot(m.params, 𝔹)

   # compute the gradient w.r.t. inputs 𝐫 in reverse mode
   ∂φ_∂𝔹 = m.params 
   _, _, ∂φ_∂Rn, ∂φ_∂Ylm = pb_𝔹(∂φ_∂𝔹)
   ∂φ_∂r = P4ML.pullback(∂φ_∂Rn, m.rbasis, r)
   ∂φ_∂𝐲 = P4ML.pullback(∂φ_∂Ylm, m.ybasis, 𝐲)

   # finally we have to transform the gradient w.r.t. r to a gradient w.r.t. 𝐫
   ∇φ = [ ∂φ_∂r[j] * (𝐫[j] / r[j]) + ∂φ_∂𝐲[j]   for j = 1:length(𝐫) ]

   return φ, ∇φ
end



## 
# CONSTRUCTION OF THE ACE MODEL 

# Some model parameters that we will use: 
Dtot = 8   # total degree; specifies the trunction of embeddings and correlations
maxl = 6    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = [ (n = n,) for n = 0:Dtot ]
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

##

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                        minn = 0, maxn = Dtot, maxl = maxl, 
                        level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                        maxlevel = Dtot)

##

# in the pre-specification we only imposed the total degree truncation, everything 
# else will be handled by the symmetrization operator within the model 
# construction; along the way we will also prune the nnll list.
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

#
# putting together everything we've construced we can now generate the model 
# here we give the model some random parameters just for testing. 
#
model = SimpleACE2(rbasis, ybasis, 𝔹basis, randn(length(𝔹basis)) )

##
# we want to check whether the model is invariant under rotations, and whether 
# the gradient is correctly implemented. 

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
rand_rot() = ( K = @SMatrix randn(3,3); exp(K - K') )

# generate a random configuration of nX points in the unit ball
nX = 7   # number of particles / points 
𝐫 = [ rand_x() for _ = 1:nX ]
Q = rand_rot() 
perm = randperm(nX)
Q𝐫 = Ref(Q) .* 𝐫[perm]

φ, ∇φ = eval_with_grad(model, 𝐫)
φQ, ∇φQ = eval_with_grad(model, Q𝐫)

# invariance of the model under rotations and permutations
@show φ ≈ φQ
# check co-variance of the gradient / forces 
@show Ref(Q) .* ∇φ[perm] ≈ ∇φQ

## check correctness of gradients 
# ForwardDiff can handle Vector{SVector}, so we have to work around that 
using ForwardDiff
_2mat(𝐱::AbstractVector{SVector{3, T}}) where {T} = collect(reinterpret(reshape, T, 𝐱))
_2vecs(X::AbstractMatrix{T}) where {T} = [ SVector{3, T}(X[:, i]) for i = 1:size(X, 2) ]

F = R -> eval_with_grad(model, _2vecs(R))[1]
∇F = R -> _2mat(eval_with_grad(model, _2vecs(R))[2])
∇F_ad = R -> ForwardDiff.gradient(F, R)

R = _2mat(𝐫)
@show ∇F(R) ≈ ∇F_ad(R)