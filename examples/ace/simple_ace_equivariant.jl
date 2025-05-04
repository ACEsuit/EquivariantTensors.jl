# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model "by hand" (as opposed to via an ML framework)
# Here we use all the utility functions that the ET library offers, 
# whereas in `simple_ace.jl` we do most steps by hand. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, LinearAlgebra, Random

##

# This struct defines a simple ACE-like model. The inputs are a cloud of points
# 𝐫 = (r₁, r₂, ..., rₙ) in 3D space. The output of the model is a scalar that 
# is invariant under rotations, reflections and permutations. 

struct SimpleACE3{T, RB, YB, BB}
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   symbasis::BB    # symmetric basis 
   params::Vector{T}   # model parameters
end

function evaluate(m::SimpleACE3, 𝐫::AbstractVector{<: SVector{3}})
   # [1] Embeddings: evaluate the Rn and Ylm embeddings
   #   Rn[j] = Rn(norm(𝐫[j])), Ylm[j] = Ylm(Rs[j])
   Rn = P4ML.evaluate(m.rbasis, norm.(𝐫))
   Ylm = P4ML.evaluate(m.ybasis, 𝐫)
   # [2] feed the Rn, Ylm embeddings through the sparse ACE model 
   𝔹 = ET.evaluate(m.symbasis, Rn, Ylm)
   # [3] the model output value is the dot product with the parameters 
   return sum(m.params .* 𝔹)
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
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

##

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 1, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

##

# in the pre-specification we only imposed the total degree truncation, everything 
# else will be handled by the symmetrization operator within the model 
# construction; along the way we will also prune the nnll list.
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 1, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

#
# putting together everything we've construced we can now generate the model 
# here we give the model some random parameters just for testing. 
#
model = SimpleACE3(rbasis, ybasis, 𝔹basis, randn(length(𝔹basis)) )

##
# we want to check whether the model is invariant under rotations, and whether 
# the gradient is correctly implemented. 

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
rand_rot() = ( K = @SMatrix randn(3,3); exp(K - K') )

# generate a random configuration of nX points in the unit ball
nX = 7   # number of particles / points 
𝐫 = [ rand_x() for _ = 1:nX ]

using WignerD, Rotations
θ = 2*π*rand(3) 
Q = Rotations.RotZYZ(θ...)
DQ = real.(ET.O3.Ctran(1) * conj.(WignerD.wignerD(1, θ...)) * ET.O3.Ctran(1)')

perm = randperm(nX)
Q𝐫 = Ref(Q) .* 𝐫[perm]

φ  = evaluate(model, 𝐫)
φQ = evaluate(model, Q𝐫)

@show DQ * φ ≈ φQ
