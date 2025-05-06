# This example is a brief demonstration how to build an ACE-like  model 
# where the output is an equivariant matrix, i.e,. 
#   φ ∘ Q = D φ D' 
# A paradigm example would be a model for a pp-block in a tight-binding 
# Hamiltonian. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random
using Zygote 

include("lineartransform.jl")
##

struct SimpleACE3{T, RB, YB, BB0, BB2}
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   symbasis0::BB0    # symmetric basis 
   symbasis2::BB2    # symmetric basis 
   params0::Vector{T}   # model parameters
   params2::Vector{T}   # model parameters
   trans_params::Vector{T} # model parameters
end

function evaluate(m::SimpleACE3, 𝐫::AbstractVector{<: SVector{3}}; basis = complex)
   # [1] Embeddings: evaluate the Rn and Ylm embeddings
   #   Rn[j] = Rn(norm(𝐫[j])), Ylm[j] = Ylm(Rs[j])
   Rn = P4ML.evaluate(m.rbasis, norm.(𝐫))
   Ylm = P4ML.evaluate(m.ybasis, 𝐫)
   # [2] feed the Rn, Ylm embeddings through the sparse ACE model 
   𝔹0 = ET.evaluate(m.symbasis0, Rn, Ylm)
   𝔹2 = ET.evaluate(m.symbasis2, Rn, Ylm)
   # [3] the model output value is the dot product with the parameters 
   y0 = sum(m.params0 .* 𝔹0)
   y2 = sum(m.params2 .* 𝔹2)
   return trans_y_pp(y0, y2, m.trans_params; basis = basis)
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
ybasis = P4ML.complex_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

##

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 2, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

##

# We now need TWO bases, one for L = 0 and one for L = 2; these can then be 
# combined into a model for the matrix φ_pp ; first we try it with the 
# complex spherical harmonics. 

𝔹basis0 = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = complex )

𝔹basis2 = ET.sparse_equivariant_tensor(; 
            L = 2, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = complex )
#
# putting together everything we've construced we can now generate the model 
# here we give the model some random parameters just for testing. 
#
model = SimpleACE3(rbasis, ybasis, 𝔹basis0, 𝔹basis2, randn(length(𝔹basis0)), randn(length(𝔹basis2)), randn(2))

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
cD = conj.(WignerD.wignerD(1, θ...))
rD = real.(ET.O3.Ctran(1) * cD * ET.O3.Ctran(1)')


perm = randperm(nX)
Q𝐫 = Ref(Q) .* 𝐫[perm]

φ  = evaluate(model, 𝐫; basis = complex)
φQ = evaluate(model, Q𝐫; basis = complex)
@show cD * φ * cD' ≈ φQ


## test for real ones 
# Now we redo the same test with real spherical harmonics. 

rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

nnll_long = ET.sparse_nnll_set(; L = 2, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

𝔹basis0 = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )
𝔹basis2 = ET.sparse_equivariant_tensor(; 
            L = 2, mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

model = SimpleACE3(rbasis, ybasis, 𝔹basis0, 𝔹basis2, 
                  randn(length(𝔹basis0)),  randn(length(𝔹basis2)), randn(2))

##
# we want to check whether the model is invariant under rotations, and whether 
# the gradient is correctly implemented. 

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
rand_rot() = ( K = @SMatrix randn(3,3); exp(K - K') )

# generate a random configuration of nX points in the unit ball
nX = 7   # number of particles / points 
𝐫 = [ rand_x() for _ = 1:nX ]

perm = randperm(nX)
Q𝐫 = Ref(Q) .* 𝐫[perm]

φ  = evaluate(model, 𝐫; basis = real)
φQ = evaluate(model, Q𝐫; basis = real)
@show rD * φ * rD' ≈ φQ
