#
# Moved some frules and pushforwards out of the way of the main files 
# because I believe them to be unnecessary. This will help 
# organize cleanup and refactoring later on.
#

# -------------------------------------- 
#    ace/sparse_ace_basis.jl
# -------------------------------------- 

# -------------- frules for forward-mode AD

import ChainRulesCore: frule

function frule((_, Δtensor, ΔRnl, ΔYlm, Δps, Δst),
               ::typeof(evaluate), tensor::SparseACEbasis, Rnl, Ylm, ps, st)
   # Forward pass with tangent computation using chain rule
   # 1. A = abasis(Rnl, Ylm)
   #    ∂A = ∂abasis/∂Rnl * ΔRnl + ∂abasis/∂Ylm * ΔYlm
   # 2. AA = aabasis(A)
   #    ∂AA = ∂aabasis/∂A * ∂A
   # 3. BB = A2Bmaps .* AA
   #    ∂BB = A2Bmaps .* ∂AA

   TA = promote_type(eltype(Rnl), eltype(Ylm))

   # Forward pass through A basis
   A = zeros(TA, length(tensor.abasis))
   evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # Compute tangent of A using frule or direct differentiation
   # For the pooled sparse product: A[i] = ∑_α Rnl[α, n] * Ylm[α, l]
   # ∂A[i] = ∑_α (ΔRnl[α, n] * Ylm[α, l] + Rnl[α, n] * ΔYlm[α, l])
   T∂A = promote_type(eltype(ΔRnl), eltype(ΔYlm), eltype(Rnl), eltype(Ylm))
   ∂A = zeros(T∂A, length(tensor.abasis))
   _pushforward_abasis!(∂A, tensor.abasis, Rnl, Ylm, ΔRnl, ΔYlm)

   # Forward pass through AA basis
   AA = zeros(TA, length(tensor.aabasis))
   evaluate!(AA, tensor.aabasis, A)

   # Compute tangent of AA using chain rule through SparseSymmProd
   # AA[i] = ∏_t A[ϕ_t(i)]
   # ∂AA[i] = AA[i] * ∑_t (∂A[ϕ_t(i)] / A[ϕ_t(i)])
   ∂AA = zeros(T∂A, length(tensor.aabasis))
   _pushforward_aabasis!(∂AA, tensor.aabasis, A, ∂A)

   # Apply coupling coefficients: BB = A2Bmaps .* AA
   BB = tensor.A2Bmaps .* Ref(AA)
   ∂BB = tensor.A2Bmaps .* Ref(∂AA)

   return BB, ∂BB
end

# Helper: pushforward through the A (pooled sparse product) basis
function _pushforward_abasis!(∂A, abasis, Rnl, Ylm, ΔRnl, ΔYlm)
   for (iA, (n, l)) in enumerate(abasis.spec)
      ∂a = zero(eltype(∂A))
      for α in axes(Rnl, 1)
         ∂a += ΔRnl[α, n] * Ylm[α, l] + Rnl[α, n] * ΔYlm[α, l]
      end
      ∂A[iA] = ∂a
   end
   return ∂A
end

# Helper: pushforward through the AA (sparse symmetric product) basis
# We compute both AA and ∂AA simultaneously to avoid redundant evaluation
function _pushforward_aabasis!(∂AA, aabasis, A, ∂A)
   num1 = aabasis.num1
   nodes = aabasis.nodes

   # We need the AA values as we go, so compute them in a temporary
   # The first num1 elements of "nodes" correspond to A[1:num1]
   TAA = eltype(A)
   AA_local = zeros(TAA, length(nodes))

   # First num1 elements are just copies of A
   for i = 1:num1
      AA_local[i] = A[i]
      ∂AA[i] = ∂A[i]
   end

   # Higher order terms use the DAG structure
   for iAA = num1+1:length(nodes)
      n1, n2 = nodes[iAA]
      # AA[iAA] = AA[n1] * AA[n2]
      AA_local[iAA] = AA_local[n1] * AA_local[n2]
      # ∂AA[iAA] = ∂AA[n1] * AA[n2] + AA[n1] * ∂AA[n2]
      ∂AA[iAA] = ∂AA[n1] * AA_local[n2] + AA_local[n1] * ∂AA[n2]
   end
   return ∂AA
end


#=
# ----------------------------------------
#  experimental pushforwards 

function _pfwd(tensor::SparseACE{T}, Rnl, Ylm, ∂Rnl, ∂Ylm) where {T}
   A, ∂A = _pfwd(tensor.abasis, (Rnl, Ylm), (∂Rnl, ∂Ylm))
   _AA, _∂AA = _pfwd(tensor.aabasis, A, ∂A)

   # project to the actual AA basis 
   proj = tensor.aabasis.projection
   AA = _AA[proj]  
   ∂AA = _∂AA[proj, :]

   # evaluate the coupling coefficients
   B = tensor.A2Bmap * AA 
   ∂B = tensor.A2Bmap * ∂AA 
   return B, ∂B 
end


function _pfwd(abasis::Polynomials4ML.PooledSparseProduct{2}, RY, ∂RY) 
   R, Y = RY 
   TA = typeof(R[1] * Y[1])
   ∂R, ∂Y = ∂RY
   ∂TA = typeof(R[1] * ∂Y[1] + ∂R[1] * Y[1])

   # check lengths 
   nX = size(R, 1)
   @assert nX == size(R, 1) == size(∂R, 1) == size(Y, 1) == size(∂Y, 1)

   A = zeros(TA, length(abasis.spec))
   ∂A = zeros(∂TA, size(∂R, 1), length(abasis.spec))

   for i = 1:length(abasis.spec)
      @inbounds begin 
         n1, n2 = abasis.spec[i]
         ai = zero(TA)
         @simd ivdep for α = 1:nX 
            ai += R[α, n1] * Y[α, n2]
            ∂A[α, i] = R[α, n1] * ∂Y[α, n2] + ∂R[α, n1] * Y[α, n2]
         end 
         A[i] = ai
      end 
   end 
   return A, ∂A
end 


function _pfwd(aabasis::Polynomials4ML.SparseSymmProdDAG, A, ∂A)
   n∂ = size(∂A, 1)
   num1 = aabasis.num1 
   nodes = aabasis.nodes 
   AA = zeros(eltype(A), length(nodes))
   T∂AA = typeof(A[1] * ∂A[1])
   ∂AA = zeros(T∂AA, length(nodes), size(∂A, 1))
   for i = 1:num1 
      AA[i] = A[i] 
      for α = 1:n∂
         ∂AA[i, α] = ∂A[α, i]
      end
   end 
   for iAA = num1+1:length(nodes)
      n1, n2 = nodes[iAA]
      AA_n1 = AA[n1]
      AA_n2 = AA[n2]
      AA[iAA] = AA_n1 * AA_n2
      for α = 1:n∂
         ∂AA[iAA, α] = AA_n2 * ∂AA[n1, α] + AA_n1 * ∂AA[n2, α]
      end
   end
   return AA, ∂AA
end


=#


