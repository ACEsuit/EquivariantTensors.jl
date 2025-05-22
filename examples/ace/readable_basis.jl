# ------------------------------------------------------------------------------
# This example demonstrates how to transfer parameters from a lower-degree model
# to a higher-degree model. 
# A paradigm example would be a model for a pp-block in a tight-binding 
# Hamiltonian. 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random
using Zygote 

struct SimpleACE4{T, RB, YB, BB, TT}
   rbasis::RB      # radial embedding Rn
   ybasis::YB      # angular embedding Ylm
   symbasis::BB    # symmetric basis -> L = 0, 2
   params0::Vector{T}   # model parameters
   params2::Vector{T}   # model parameters
   trans::TT
end

function evaluate(m::SimpleACE4, 𝐫::AbstractVector{<: SVector{3}}; basis = complex)
   # [1] Embeddings: evaluate the Rn and Ylm embeddings
   #   Rn[j] = Rn(norm(𝐫[j])), Ylm[j] = Ylm(Rs[j])
   Rn = P4ML.evaluate(m.rbasis, norm.(𝐫))
   Ylm = P4ML.evaluate(m.ybasis, 𝐫)
   # [2] feed the Rn, Ylm embeddings through the sparse ACE model 
   𝔹0, 𝔹2, = ET.evaluate(m.symbasis, Rn, Ylm)
   # [3] the model output value is the dot product with the parameters 
   y0 = sum(m.params0 .* 𝔹0)
   y2 = sum(m.params2 .* 𝔹2)
   y = ET.O3.yvector(y0, nothing, y2)
   return model.trans(y)
end

## 
# CONSTRUCTION OF THE ACE MODEL 

# Some model parameters that we will use: 
Dtot = 8   # total degree; specifies the trunction of embeddings and correlations
maxl = 6    # maximum degree of spherical harmonics 
ORD = 2     # correlation-order (body-order = ORD + 1)

##
# first specify the radial and angular embeddings 
rbasis_old = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis_old) 
ybasis_old = P4ML.complex_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis_old)

##

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 2, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

##
𝔹basis_old = ET.sparse_equivariant_tensors(; 
            LL = (0, 2), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = complex )

model_old = SimpleACE4(rbasis_old, ybasis_old, 𝔹basis_old, 
                   randn(length(𝔹basis_old, 0)), randn(length(𝔹basis_old, 2)), 
                   ET.O3.TYVec2YMat(1, 1; basis=complex))

# -------------------------- New Model with Higher Degree --------------------------
Dtot = 10   # total degree; specifies the trunction of embeddings and correlations
maxl = 8    # maximum degree of spherical harmonics 
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

𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = (0, 2), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = complex )

model = SimpleACE4(rbasis, ybasis, 𝔹basis, 
                   randn(length(𝔹basis, 0)), randn(length(𝔹basis, 2)), 
                   ET.O3.TYVec2YMat(1, 1; basis=complex))

# -------------------------- Parameter Transfer --------------------------
# Transfer L=0 parameters
NNLL = ET.get_nnll_spec(𝔹basis, 1)
NNLL_old = ET.get_nnll_spec(𝔹basis_old, 1)
model.params0 .= 0.0
_map = ET.dict_invmap(NNLL)
for (idx, t) in enumerate(NNLL_old)
    model.params0[_map[t]] = model_old.params0[idx]
end

# Transfer L=2 parameters
NNLL = ET.get_nnll_spec(𝔹basis, 2)
NNLL_old = ET.get_nnll_spec(𝔹basis_old, 2)
model.params2 .= 0.0
_map = ET.dict_invmap(NNLL)
for (idx, t) in enumerate(NNLL_old)
    model.params2[_map[t]] = model_old.params2[idx]
end

# -------------------------- Evaluation Example --------------------------
rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()

# generate a random configuration of nX points in the unit ball
nX = 7   # number of particles / points 
𝐫 = [ rand_x() for _ = 1:nX ]

# Evaluate the two models on the same configuration
φ1  = evaluate(model, 𝐫; basis = complex)
φ2  = evaluate(model_old, 𝐫; basis = complex)

# Assert that outputs match (after parameter transfer)
@assert φ1 ≈ φ2