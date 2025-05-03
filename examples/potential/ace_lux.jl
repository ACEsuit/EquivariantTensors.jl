
# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model within the Lux framework. This is less performance 
# than "manual" model building but allows for faster prototyping and 
# experimentation with different model architectures. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random
using Zygote, LuxCore, Lux

## 
# CONSTRUCTION OF THE ACE MODEL 
# The first few steps are the same as in `simple_ace.jl`, we need to build the 
# radial and angular embeddings, and then the 𝔹 basis layer. 

# Some model parameters that we will use: 
Dtot = 8   # total degree; specifies the trunction of embeddings and correlations
maxl = 6    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

# in the pre-specification we only imposed the total degree truncation, everything 
# else will be handled by the symmetrization operator within the model 
# construction; along the way we will also prune the nnll list.
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

##
# Now, instead of a custom made model struct, we just use a Lux Chain to 
# build the model. 

model = Chain(
      Parallel(nothing, 
               Chain( WrappedFunction(𝐫 -> norm.(𝐫)),  
                      P4ML.lux(rbasis) ), 
               P4ML.lux(ybasis)),
      𝔹basis, 
      Dense(length(𝔹basis) => 1), 
      WrappedFunction(x -> x[1])
      )

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

rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)
φ, _ = Lux.apply(model, 𝐫, ps, st)
φQ, _ = Lux.apply(model, Q𝐫, ps, st)

# invariance of the model under rotations and permutations
@show φ ≈ φQ

## 
# To compute gradients we can again construct a convenience function that 
# evaluates the model and then the gradient in a backward pass. 

# NB: this isn't working yet, apparently we are mutating some array outside 
#     of a custom rrule. 
#=

function ace_with_grad(m, 𝐫::AbstractVector{<: SVector{3}}, ps, st)
   φ, (∇φ,) = Zygote.withgradient(x -> Lux.apply(model, x, ps, st), 𝐫)
   return φ, ∇φ
end

φ, ∇φ = ace_with_grad(model, 𝐫, ps, st)
φQ, ∇φQ = eval_with_grad(model, Q𝐫)

# invariance of the model under rotations and permutations
@show φ ≈ φQ
# check co-variance of the gradient / forces 
@show Ref(Q) .* ∇φ[perm] ≈ ∇φQ

check correctness of gradients 
# ForwardDiff can handle Vector{SVector}, so we have to work around that 
using ForwardDiff
_2mat(𝐱::AbstractVector{SVector{3, T}}) where {T} = collect(reinterpret(reshape, T, 𝐱))
_2vecs(X::AbstractMatrix{T}) where {T} = [ SVector{3, T}(X[:, i]) for i = 1:size(X, 2) ]

F = R -> evaluate(model, _2vecs(R))
∇F = R -> _2mat(eval_with_grad(model, _2vecs(R))[2])
∇F_ad = R -> ForwardDiff.gradient(F, R)

R = _2mat(𝐫)
@show ∇F(R) ≈ ∇F_ad(R)

=#
