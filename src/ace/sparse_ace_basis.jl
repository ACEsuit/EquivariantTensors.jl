
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: mul!
import ChainRulesCore: NoTangent, rrule, ZeroTangent
import LuxCore: AbstractLuxLayer, initialparameters, initialstates, apply 


struct SparseACEbasis{NL, TA, TAA, TSYM} <: AbstractLuxLayer
   abasis::TA
   aabasis::TAA
   A2Bmaps::TSYM
   LL::NTuple{NL, Int} 
   lens::NTuple{NL, Int} 
   # ---- 
   meta::Dict{String, Any}
end

function SparseACEbasis(abasis, aabasis, A2Bmaps, meta) 
   LL = []
   lens = [] 
   for i = 1:length(A2Bmaps)
      tLp1 = length(A2Bmaps[i][1])
      push!(LL, (tLp1 - 1) ÷ 2)
      push!(lens, size(A2Bmaps[i], 1))
   end
   SparseACEbasis(abasis, aabasis, A2Bmaps, 
             tuple(LL...), tuple(lens...), meta)
end

Base.length(tensor::SparseACEbasis) = sum(tensor.lens)

function Base.length(tensor::SparseACEbasis, L::Integer)
   for (il, l) in enumerate(tensor.LL)
      if l == L
         return tensor.lens[il]
      end
   end
   error("Layer does not have an for L = $L output")
end

function Base.show(io::IO, l::SparseACEbasis)
   print(io, "SparseACEbasis(L = $(l.LL))")
end


# ----------------------------------------
# Lux integration 

(l::SparseACEbasis)(BB::Tuple, ps, st) = evaluate(l, BB..., ps, st), st 

initialparameters(rng::AbstractRNG, bas::SparseACEbasis) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, bas::SparseACEbasis) =
         ( aspec = bas.abasis.spec,
            aaspecs = bas.aabasis.specs,
            A2Bmaps = SparseMatCSX.(bas.A2Bmaps), )


# ----------------------------------------
# evaluation kernels 

#=
function evaluate!(B, tensor::SparseACEbasis{T}, Rnl, Ylm) where {T}
   # evaluate the A basis
   TA = promote_type(eltype(Rnl), eltype(Ylm))
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

function whatalloc(::typeof(evaluate!), tensor::SparseACEbasis, Rnl, Ylm)
   TA = promote_type(eltype(Rnl), eltype(Ylm))
   TB = _promote_mul_type(TA, eltype(tensor.A2Bmap))
   return TB, length(tensor)
end

=#

evaluate(tensor::SparseACEbasis, Rnl, Ylm) = 
      evaluate(tensor, Rnl, Ylm, NamedTuple(), NamedTuple()) 

#=
function evaluate(tensor::SparseACEbasis, Rnl, Ylm, ps, st)
   allocinfo = whatalloc(evaluate!, tensor, Rnl, Ylm)
   B = zeros(allocinfo...)
   return evaluate!(B, tensor, Rnl, Ylm)
end
=#

function evaluate(tensor::SparseACEbasis, Rnl, Ylm, ps, st)
   A = ka_evaluate(tensor.abasis, (Rnl, Ylm))
   AA = ka_evaluate(tensor.aabasis, A)
   # evaluate the coupling coefficients
   BB = tensor.A2Bmaps .* Ref(AA)
   return BB
end 


# for Ten3 inputs, there is no CPU implementation 
# so everything based on TupTen3 gets automatically dispatched to 
# ka_evaluate, which is good. 

evaluate(tensor::SparseACEbasis, BB::TupTen3, args...) = 
      ka_evaluate(tensor, BB, args...)


evaluate(tensor::SparseACEbasis, 
         Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, 
         args...) where {T} = 
      ka_evaluate(tensor, Rnl, Ylm, args...)[1] 


# ---------


function pullback!(∂Rnl, ∂Ylm, 
                   ∂BB, tensor::SparseACEbasis, Rnl, Ylm, A)

   @no_escape begin 
   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                           
   # ∂Ei / ∂AA = ∂Ei / ∂B * ∂B / ∂AA = (WB[i_z0]) * A2Bmap
   # ∂AA = tensor.A2Bmap' * ∂B   
   # T_∂AA = promote_type(eltype(∂B), eltype(tensor.A2Bmap))
   # ∂AA = @alloc(T_∂AA, size(tensor.A2Bmap, 2))
   # mul!(∂AA, tensor.A2Bmap', ∂B)
   # ∂AA = tensor.A2Bmap' * ∂B
   # T_∂AA = eltype(∂AA)
   # Dexuan's draft: 
   #  for (i, ∂Bᵢ) in enumerate(∂BB)
   #      ∂AA .+= tensor.A2Bmaps[i]' * ∂Bᵢ
   #  end   
   ∂AA = sum( tensor.A2Bmaps[i]' * ∂BB[i] 
              for i = 1:length(∂BB) )
   T_∂AA = eltype(∂AA)

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
                   ∂BB, tensor::SparseACEbasis, Rnl, Ylm
                   )
   # TODO: may need to check the type of ∂BB too, but this is a bit 
   #       tricky because of the SVectors that can be in there...
   TB = eltype.(eltype.(∂BB))
   TA = promote_type(eltype(Rnl), eltype(Ylm), TB...)
   return (TA, size(Rnl)...), (TA, size(Ylm)...)
end

function pullback(∂BB, tensor::SparseACEbasis{T}, Rnl, Ylm, A) where {T}
   alc_∂Rnl, alc_∂Ylm = whatalloc(pullback!, ∂BB, tensor, Rnl, Ylm)
   ∂Rnl = zeros(alc_∂Rnl...)
   ∂Ylm = zeros(alc_∂Ylm...)
   return pullback!(∂Rnl, ∂Ylm, ∂BB, tensor, Rnl, Ylm, A)
end


# ChainRules integration 
using ChainRulesCore: unthunk 

function rrule(::typeof(evaluate), tensor::SparseACEbasis, 
               Rnl::AbstractMatrix, Ylm::AbstractMatrix, ps, st)
   @info("wrong rrule")
   # evaluate the A basis
   # TA = promote_type(eltype(Rnl), eltype(eltype(Ylm)))
   # A = zeros(TA, length(tensor.abasis))    # use Bumper here
   A = evaluate(tensor.abasis, (Rnl, Ylm))

   # evaluate the AA basis
   # AA = zeros(TA, length(tensor.aabasis))     # use Bumper here
   AA = evaluate(tensor.aabasis, A)

   # evaluate the coupling coefficients
   BB = tensor.A2Bmaps .* Ref(AA)

   function pb(∂BB)
      ∂Rnl, ∂Ylm = pullback(unthunk.(∂BB), tensor, Rnl, Ylm, A)
      return NoTangent(), NoTangent(), ∂Rnl, ∂Ylm, ZeroTangent(), NoTangent() 
   end
   return BB, pb
end

# rrule for 3D array inputs (batched evaluation) - delegates to ka_evaluate
function rrule(::typeof(evaluate), tensor::SparseACEbasis,
               Rnl::AbstractArray{T, 3}, Ylm::AbstractArray{T, 3}, 
               ps, st) where {T}
   # Delegate to ka_evaluate which has its own rrule
   𝔹, A, 𝔸 = _ka_evaluate(tensor, Rnl, Ylm,
                          st.aspec, st.aaspecs, st.A2Bmaps)

   function pb_3d(∂out)
      ∂𝔹 = ∂out[1]
      ∂Rnl, ∂Ylm = _ka_pullback(∂𝔹, tensor, Rnl, Ylm,
                                A, 𝔸,
                                st.aspec, st.aaspecs,
                                st.A2Bmaps)
      return NoTangent(), NoTangent(), ∂Rnl, ∂Ylm,
             NoTangent(), NoTangent()
   end

   return (𝔹, st), pb_3d
end


# --------------------------------------------------------
# 
#  Jacobian of basis w.r.t. inputs (normally positions) 
#
# Assume the input data is organized as follows: 
#   Rnl : #j x #i x #R  array with #R the length of the radial basis 
#   Ylm : #j x #i x #Y  array with #Y the length of the spherical basis
#   dRnl, dYlm : same shape as Rnl, Ylm with 
#       dRnl[j, i, k] = ∂Rnl[i, j] / ∂X[i, j]  
#       dYlm[j, i, k] = ∂Ylm[i, j] / ∂X[i, j]
#

function _jacobian_X(tensor::SparseACEbasis, 
                     Rnl, Ylm, 
                     dRnl, dYlm, 
                     ps, st)
   backend = KernelAbstractions.get_backend(Rnl)
   KernelAbstractions.synchronize(backend)

   A, ∂A = _jacobian_X(tensor.abasis, (Rnl, Ylm), (dRnl, dYlm),
                       st.aspec)
   KernelAbstractions.synchronize(backend)

   AA, ∂AA = _jacobian_X(tensor.aabasis, A, ∂A,
                         st.aaspecs)

   KernelAbstractions.synchronize(backend)
   
   # BB = tensor.A2Bmap * AA  if vector (single input)
   #     or AA * A2Bmap'  if matrix (batch)
   # BB = #nodes x #features 
   # ∂BB = maxneigs x #nodes x #features
   # for now assume only one basis ... 
   @assert length(st.A2Bmaps) == 1 "Jacobian currently only supports single basis"
   A2Bmaps1 = st.A2Bmaps[1] 
   𝔹 = permutedims( mul(A2Bmaps1, permutedims(AA) ) )

   # convert 3-tensor to matrix, apply A2Bmaps, then back to 3-tensor
   # this should be merged into a single kernel for efficiency 
   ∂AA_mat = reshape(∂AA, :, size(∂AA, 3))
   ∂𝔹_mat1 = permutedims( mul(A2Bmaps1, permutedims(∂AA_mat)) )

   ∂𝔹 = reshape(∂𝔹_mat1, size(∂AA, 1), :, size(∂𝔹_mat1, 2))

   return (𝔹,), (∂𝔹,)
end


# --------------------------------------------------------


const NT_NL_SPEC = NamedTuple{(:n, :l), Tuple{Int, Int}}

_nl(bb) = [(n = b.n, l = b.l) for b in bb]

function get_nnll_spec(tensor::SparseACEbasis{NL, TA, TAA, TSYM}, idx) where {NL, TA, TAA, TSYM}
   spec = tensor.meta["𝔸spec"]::Vector{Vector{@NamedTuple{n::Int, l::Int, m::Int}}}
   A2Bmap = tensor.A2Bmaps[idx]
   nBB = size(A2Bmap, 1)
   nnll_list = Vector{Vector{NT_NL_SPEC}}(undef, nBB)
   for i in 1:nBB
      AAidx_nnz = A2Bmap[i, :].nzind
      bbs = spec[AAidx_nnz]
      nnll_list[i] = _nl(bbs[1])
   end
   return nnll_list
end

