# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model "by hand" (as opposed to via an ML framework). 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random

##

# This struct defines a simple ACE-like model. The inputs are a cloud of points
# 𝐫 = (r₁, r₂, ..., rₙ) in 3D space. The output of the model is a scalar that 
# is invariant under rotations, reflections and permutations. 

struct SimpleACE{T, RB, YB, AB, AAB, BB}
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   abasis::AB      # atomic basis (pooling of Rn Ylm)
   aabasis::AAB    # n-correlations 
   symm::BB        # symmetrization
   params::Vector{T}   # model parameters
end

function eval_with_grad(m::SimpleACE, 𝐫::AbstractVector{<: SVector{3}})
   # evaluate the Rn and Ylm embeddings
   #   Rn[j] = Rn(norm(𝐫[j])), Ylm[j] = Ylm(Rs[j])
   r = norm.(𝐫)
   𝐲 = 𝐫
   Rn = P4ML.evaluate(m.rbasis, r)
   Ylm = P4ML.evaluate(m.ybasis, 𝐲)
   # evaluate the atomic basis:    A_nlm = ∑_j Rn[j] * Ylm[j]
   A = m.abasis((Rn, Ylm))
   # evaluate the n-correlations:  𝔸_𝐧𝐥𝐦 = ∏_t A_nₜlₜmₜ
   𝔸 = m.aabasis(A)
   # symmetrize the output:        𝔹 = C * 𝔸    
   𝔹 = m.symm * 𝔸
   
   # the model output value is the dot product with the parameters 
   φ = dot(m.params, 𝔹)

   # compute the gradient w.r.t. inputs 𝐫 in reverse mode
   ∂φ_∂𝔹 = m.params 
   ∂φ_∂𝔸 = m.symm' * ∂φ_∂𝔹
   ∂φ_∂A = ET.pullback(∂φ_∂𝔸, m.aabasis, A)
   ∂φ_∂Rn, ∂φ_∂Ylm = ET.pullback(∂φ_∂A, m.abasis, (Rn, Ylm))
   ∂φ_∂r = P4ML.pullback(∂φ_∂Rn, m.rbasis, r)
   ∂φ_∂𝐲 = P4ML.pullback(∂φ_∂Ylm, m.ybasis, 𝐲)

   # finally we have to transform the gradient w.r.t. r to a gradient w.r.t. 𝐫
   ∇φ = [ ∂φ_∂r[j] * (𝐫[j] / r[j]) + ∂φ_∂𝐲[j]   for j = 1:length(𝐫) ]

   return φ, ∇φ
end


## 
# CONSTRUCTION OF THE ACE MODEL 

# Some model parameters that we will use: 
Dtot = 7   # total degree; specifies the trunction of embeddings and correlations
maxL = 5    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# [1] first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1) 
Rn_spec = [ (n = n,) for n = 0:Dtot ]
ybasis = P4ML.real_sphericalharmonics(maxL)
Ylm_spec = P4ML.natural_indices(ybasis), 

# generate the nnll basis pre-specification
nnll_long = let Dtot = Dtot, maxL = maxL 
   nl = [ (n=n, l=l) for n = 0:Dtot for l = 0:maxL if (n + l <= Dtot) ]
   comb = with_replacement_combinations(0:length(nl), ORD)
   ii2bb = ii -> eltype(nl)[ nl[i] for i in ii[ii .> 0] ]
   myfilter = ii -> ( bb = ii2bb(ii); 
                  ( length(bb) > 0 && 
                    sum(b.n + b.l for b in bb; init=0) <= Dtot && 
                    iseven(sum(b.l for b in bb; init=0)) ) ) 
   return [ ii2bb(ii) for ii in comb if myfilter(ii) ]
end


##
# in the pre-specification we only imposed the total degree truncation, everything 
# else will be handled by the symmetrization operator within the model 
# construction; along the way we will also prune the nnll list.
ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real ) 


##

# abasis = ET.PooledSparseProduct(Aspec)
# @assert abasis.spec == Aspec
# aabasis = ET.SparseSymmProd(𝔸spec)

##
# [4] symmetrization
# the symmetrization operator 𝔸 ↦ 𝔹 = 𝒞 ⋅ 𝔸 requires some information about 
# the basis functions that we now have to reconstruct from the specification of 
# the 𝔸, A, R, Y layers. It basically means rewriting 𝔸spec in a format that  
# identifies the n, l, m channels. Luckily we already have this in the form of 
# the `ii2bb` function.
nnllmm = [ ii2bb(ii) for ii in 𝔸spec ]

# this function creates a unique way to lookup permutation-invariant features
function bb_key(nn, ll, mm) 
   return sort( [ (nn[α], ll[α], mm[α]) for α = 1:length(nn) ] )
end 

inv_nnllmm = Dict( bb_key(bb...) => i for (i, bb) in enumerate(nnllmm) )

# from this we can extract all unique (nn, ll) blocks (the mm will just be used 
# in generating the coupled / symmetrized basis functions)
nnll = unique( [(nn, ll) for (nn, ll, mm) in nnllmm] )

# Now for each (nn, ll) block we can generate all possible invariant basis 
# functions. We assemble the symmetrization operator in triplet format, 
# which can conveniently account for double-counting of entries.
irow = Int[]; jcol = Int[]; val = Float64[]

num𝔹 = 0 
for (i, (nn, ll)) in enumerate(nnll)
   cc, MM = ET.O3.coupling_coeffs(0, ll, nn; PI = true, basis = real)
   num_b = size(cc, 1)   # number of invariant basis functions for this block 
   # lookup the corresponding (nn, ll, mm) in the 𝔸 specification 
   idx_𝔸 = [inv_nnllmm[bb_key(nn, ll, mm)] for mm in MM] 
   for q = 1:num_b 
      num𝔹 += 1
      for j = 1:length(idx_𝔸)
         push!(irow, num𝔹); push!(jcol, idx_𝔸[j]); push!(val, cc[q, j])
      end
   end
end

# we can now generate the symmetrization operator by concatenating the 
# sparse coupling vectors stored in 𝒞. 
symm = sparse(irow, jcol, val, num𝔹, length(𝔸spec)) 
@show num𝔹

##
# putting together everything we've construced we can now generate the model 
# here we give the model some random parameters just for testing. 
#
model = SimpleACE(rbasis, ybasis, abasis, aabasis, symm, randn(num𝔹) )

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
