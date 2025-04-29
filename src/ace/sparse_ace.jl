
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: mul!
import ChainRulesCore: NoTangent, rrule

struct SparseACE{T, TA, TAA}
   abasis::TA
   aabasis::TAA
   A2Bmap::SparseMatrixCSC{T, Int}
   # ---- 
   meta::Dict{String, Any}
end

Base.length(tensor::SparseACE) = size(tensor.A2Bmap, 1) 


function evaluate!(B, tensor::SparseACE{T}, Rnl, Ylm) where {T}
   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl), eltype(eltype(Ylm)))
   A = zeros(TA, length(tensor.abasis))    # use Bumper here
   evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   evaluate!(AA, tensor.aabasis, A)

   # evaluate the coupling coefficients
   # B = tensor.A2Bmap * AA
   mul!(B, tensor.A2Bmap, AA)   

   return B
end

function whatalloc(::typeof(evaluate!), tensor::SparseACE, Rnl, Ylm)
   TA = promote_type(eltype(Rnl), eltype(eltype(Ylm)))
   TB = promote_type(TA, eltype(tensor.A2Bmap))
   return TB, length(tensor)
end

function evaluate(tensor::SparseACE, Rnl, Ylm)
   allocinfo = whatalloc(evaluate!, tensor, Rnl, Ylm)
   B = zeros(allocinfo...)
   return evaluate!(B, tensor, Rnl, Ylm)
end


# ---------


function pullback!(∂Rnl, ∂Ylm, 
                   ∂B, tensor::SparseACE, Rnl, Ylm, A)

   @no_escape begin 
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                           
   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   # ∂AA = tensor.A2Bmap' * ∂B   
   T_∂AA = promote_type(eltype(∂B), eltype(tensor.A2Bmap))
   ∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
   mul!(∂AA, tensor.A2Bmap', ∂B)
   # ∂AA = tensor.A2Bmap' * ∂B
   # T_∂AA = eltype(∂AA)

   # ∂Ei / ∂A = ∂Ei / ∂AA * ∂AA / ∂A = pullback(aabasis, ∂AA)
   T_∂A = promote_type(T_∂AA, eltype(A))
   ∂A = @alloc(T_∂A, length(tensor.abasis))
   pullback!(∂A, ∂AA, tensor.aabasis, A)
   
   # ∂Ei / ∂Rnl, ∂Ei / ∂Ylm = pullback(abasis, ∂A)
   pullback!((∂Rnl, ∂Ylm), ∂A, tensor.abasis, (Rnl, Ylm))

   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end # no_escape

   return ∂Rnl, ∂Ylm
end

function whatalloc(::typeof(pullback!),  
                   ∂B, tensor::SparseACE{T}, Rnl, Ylm
                   ) where {T} 
   TA = promote_type(T, eltype(∂B), eltype(Rnl), eltype(eltype(Ylm)))
   return (TA, size(Rnl)...), (TA, size(Ylm)...)
end

function pullback(∂B, tensor::SparseACE{T}, Rnl, Ylm, A) where {T} 
   alc_∂Rnl, alc_∂Ylm = whatalloc(pullback!, ∂B, tensor, Rnl, Ylm)
   ∂Rnl = zeros(alc_∂Rnl...)
   ∂Ylm = zeros(alc_∂Ylm...)
   return pullback!(∂Rnl, ∂Ylm, ∂B, tensor, Rnl, Ylm, A)
end


# ChainRules integration 
using ChainRulesCore: unthunk 

function rrule(::typeof(evaluate), tensor::SparseACE{T}, Rnl, Ylm) where {T}

   # evaluate the A basis
   TA = promote_type(T, eltype(Rnl), eltype(eltype(Ylm)))
   A = zeros(TA, length(tensor.abasis))    # use Bumper here
   evaluate!(A, tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   evaluate!(AA, tensor.aabasis, A)

   # evaluate the coupling coefficients
   B = tensor.A2Bmap * AA

   function pb(∂B)
      ∂Rnl, ∂Ylm = pullback(unthunk(∂B), tensor, Rnl, Ylm, A)
      return NoTangent(), NoTangent(), ∂Rnl, ∂Ylm
   end
   return B, pb
end


#=

# ----------------------------------------
#  utilities 

"""
Get the specification of the BBbasis as a list (`Vector`) of vectors of `@NamedTuple{n::Int, l::Int}`.

### Parameters 

* `tensor` : a SparseACE, possibly from ACEModel
"""
function get_nnll_spec(tensor::SparseACE{T}) where {T}
   _nl(bb) = [(n = b.n, l = b.l) for b in bb]
   # assume the new ACE model NEVER has the z channel
   spec = tensor.aabasis.meta["AA_spec"]
   nBB = size(tensor.A2Bmap, 1)
   nnll_list = Vector{NT_NL_SPEC}[]
   for i in 1:nBB
      AAidx_nnz = tensor.A2Bmap[i, :].nzind
      bbs = spec[AAidx_nnz]
      @assert all([bb == _nl(bbs[1]) for bb in _nl.(bbs)])
      push!(nnll_list, _nl(bbs[1]))
   end
   @assert length(nnll_list) == nBB
   return nnll_list
end



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
