# This example is a brief demonstration how to build an ACE-like 
# O(3)-invariant model "by hand" (as opposed to via an ML framework). 

module ACE0 

import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays

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
   #   Rn[j] = Rn(norm(Rs[j])), Ylm[j] = Ylm(Rs[j])
   r = norm.(Rs)
   𝐲 = 𝐫
   Rn   = P4ML.evaluate(m.rbasis, r)
   Ylm = P4ML.evaluate(m.ybasis, 𝐲)
   # evaluate the atomic basis
   #   A_nlm = ∑_j Rn[j] * Ylm[j]
   A = m.abasis((Rn, Ylm))
   # evaluate the n-correlations
   #   𝔸_𝐧𝐥𝐦 = ∏_t A_nₜlₜmₜ
   𝔸 = m.aabasis(A)
   # symmetrize the output
   #   𝔹 = C * 𝔸    
   𝔹 = m.symm * 𝔸
   
   # the model is given by the dot product with the parameters 
   φ = dot(m.params, 𝔹)

   # compute the gradient w.r.t. inputs Rs via backpropagation 
   ∂φ_∂𝔹 = m.params 
   ∂φ_∂𝔸 = m.symm' * ∂φ_∂𝔹
   ∂φ_∂A = ET.pullback(∂φ_∂𝔸, m.aabasis, A)
   ∂φ_∂Rn, ∂φ_∂Ylm = ET.pullback(∂φ_∂A, m.abasis, (Rn, Ylm))
   ∂φ_∂r = P4ML.pullback(∂φ_∂Rn, m.rbasis, r)
   ∂φ_∂𝐲 = P4ML.pullback(∂φ_∂Ylm, m.ybasis, 𝐲)

   # finally we have to transform the gradient w.r.t. r to a gradient w.r.t. 𝐫
   ∇φ = zeros(SVector{3, T}, length(𝐫))
   for j = 1:length(𝐫)
      ∇φ[j] = ∂φ_∂r[j] * (𝐫[j] / r[j]) + ∂φ_∂𝐲[j]
   end

   return φ, ∇φ
end

end


# to construct the ACE model we 

Rn = legendre_basis(totdeg)
Ylm = RYlmBasis(maxL)
ν = 2

# Pooling and SparseProduct + n-corr 
spec1p = [(i, y) for i = 1:totdeg for y = 1:maxL]
bA = P4ML.PooledSparseProduct(spec1p)

# define n-corr spec
tup2b = vv -> [ spec1p[v] for v in vv[vv .> 0]  ]
admissible = bb -> ((length(bb) == 0) || (sum(b[1] - 1 for b in bb ) < totdeg)) # cannot use <= since we cannot approxiate poly basis corresponding to (2, 15) with (15)
filter = bb -> (length(bb) == 0 || sum(idx2lm(b[2])[1] for b in bb) <= maxL)
specAA = gensparse(; NU = ν, tup2b = tup2b, admissible = admissible, filter = filter, minvv = fill(0, ν), maxvv = fill(length(spec1p), ν), ordered = true)
spec = [ vv[vv .> 0] for vv in specAA if !(isempty(vv[vv .> 0]))]
