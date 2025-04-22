# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model "by hand" (as opposed to via an ML framework). 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random

##
module ACE0 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra

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

function eval_with_grad(m::SimpleACE, 𝐫::AbstractVector{<: SVector{3}}) where {T} 
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

end


## 
# CONSTRUCTION OF THE ACE MODEL 

# Some model parameters that we will use: 
Dtot = 6   # total degree; specifies the trunction of embeddings and correlations
maxL = 4    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# [1] first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1)
ybasis = P4ML.real_sphericalharmonics(maxL)

##
# [2] Pooling and SparseProduct
# this layer takes the embeddings of the individual particles and pools them 
# to embed the entire set of particles. (point cloud) Note this is a sparse 
# operation; only the basis functions Aₙₗₘ are computed for which n + l ≤ Dtot.
#
Aspec = [ (n+1, P4ML.lm2idx(l, m)) 
           for n = 0:Dtot for l = 0:maxL for m = -l:l if (n + l <= Dtot) ]
abasis = ET.PooledSparseProduct(Aspec)
@assert abasis.spec == Aspec

##
# [3] n-correlations 
# generating sparse n-correlations is a little more involved, and here is it 
# better to just automate this. But for a very small model we can still do it 
# by hand. 
# first get all possible combinations of A basis functions, then we will filter 
comb1 = with_replacement_combinations(0:length(Aspec), ORD)
ii2bb = ii -> begin 
      bb = [ Aspec[i] for i in ii[ii .> 0]  ];
      nn = Int[b[1]-1 for b in bb]; 
      ll = Int[P4ML.idx2lm(b[2])[1] for b in bb];
      mm = Int[P4ML.idx2lm(b[2])[2] for b in bb];
      return nn, ll, mm 
   end
myfilter = ii -> begin 
      nn, ll, mm = ii2bb(ii);
      return ( (sum(nn + ll; init=0) <= Dtot) &&  # total degree trunction
               iseven(sum(ll; init=0)) &&         # reflection-invariance
               (sum(mm; init=0) == 0)  &&         # rotation-invariance
               sum(ii) > 0 )           # drop 0-corr sure to bug 
   end 

@show length(comb1)
comb2 = [ ii for ii in comb1 if myfilter(ii) ]
@show length(comb2) 

# notice the incredible reduction in the number of features due to imposing 
# the filters given by the O(3) invariance constraints and the sparsification
# (the latter can be thought of as a smoothness prior)

# to finish the 𝔸spec we need to convert to 0-corr, 1-corr, 2-corr and 3-corr
# by dropping the zeros from the combinations 
𝔸spec = [ filter(!iszero, ii) for ii in comb2 ]
# and now we can finally generate the n-correlations layer 
aabasis = ET.SparseSymmProd(𝔸spec)

##
# [4] symmetrization
# the symmetrization operator 𝔸 ↦ 𝔹 = 𝒞 ⋅ 𝔸 requires some information about 
# the basis functions that we now have to reconstruct from the specification of 
# the 𝔸, A, R, Y layers. It basically means rewriting 𝔸spec in a "readable" 
# format. Luckily we already have this in the form of the `ii2bb` function. 
# from that we generate the "readable" spec of 𝔸 and then a lookup table.
nnllmm = [ ii2bb(ii) for ii in 𝔸spec ]

# this function creates a unique way to lookup permutation-invariant features
function sort_bb(nn, ll, mm) 
   aa = sort( [ [nn[α], ll[α], mm[α] ] for α = 1:length(nn) ] ) 
   return ntuple(i -> [ aa[α][i] for α = 1:length(nn) ], 3)
end 

inv_nnllmm = Dict( sort_bb(bb...) => i for (i, bb) in enumerate(nnllmm) )

# from this we can extract all unique (nn, ll) blocks (the mm will just be used 
# in generating the coupled / symmetrized basis functions)
nnll = unique( [(nn, ll) for (nn, ll, mm) in nnllmm] )

# Now for each (nn, ll) block we can generate all possible invariant basis 
# functions. 
𝒞 = SparseVector{Float64, Int64}[]
for (nn, ll) in nnll 
   cc, MM = ET.O3.coupling_coeffs(0, ll, nn; PI = true)
   num_b = size(cc, 1)   # number of invariant basis functions for this block 
   # lookup the corresponding (nn, ll, mm) in the 𝔸 specification 
   idx_𝔸 = [inv_nnllmm[sort_bb(nn, ll, mm)] for mm in MM] 
   for q = 1:num_b 
      push!(𝒞, SparseVector(length(𝔸spec), idx_𝔸, cc[q, :]))
   end
end

# we can now generate the symmetrization operator by concatenating the 
# sparse coupling vectors stored in 𝒞. 
symm = transpose(reduce(hcat, 𝒞))

##
# putting together everything we've construced we can now generate the model 
# here we give the model some random parameters just for testing. 
#
model = ACE0.SimpleACE(
   rbasis, ybasis, abasis, aabasis, symm, randn(length(𝒞)) )

# we want to check whether the model is invariant under rotations, and whether 
# the gradient is correctly implemented. 

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
rand_rot() = ( K = @SMatrix randn(3,3); exp(K - K') )

# generate a random configuration of nX points in the unit ball
nX = 7   # number of particles / points 
𝐫 = [ rand_x() for _ = 1:nX ]

φ, ∇φ = ACE0.eval_with_grad(model, 𝐫)

Q = rand_rot() 
Q𝐫 = Ref(Q) .* shuffle(𝐫)
φQ, ∇φQ = ACE0.eval_with_grad(model, Q𝐫)

@show φ
@show φQ