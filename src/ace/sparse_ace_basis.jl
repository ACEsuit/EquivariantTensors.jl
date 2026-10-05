
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros
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

# Underlying scalar element type of a (possibly nested / SVector-valued) array
# type. Pure type-level recursion, so it is fully inferrable.
_scalar_eltype(::Type{T}) where {T <: Number} = T
_scalar_eltype(::Type{<:AbstractArray{T}}) where {T} = _scalar_eltype(T)

# Promote the scalar eltype across the cotangent blocks ∂BB, type-stably for both
# a `Tuple` of blocks (possibly heterogeneous, e.g. mixed L=0 / L=1) and a
# homogeneous `AbstractArray` of blocks.
_blocks_scalar_eltype(∂BB::Tuple) =
      reduce(promote_type, map(b -> _scalar_eltype(typeof(b)), ∂BB))
# Homogeneous (concrete-eltype) arrays take the inferrable type-level path; a
# heterogeneous container (e.g. Vector{Any}) falls back to a per-element reduction
# (not type-stable, but such inputs are degenerate — blocks normally come as a Tuple).
_blocks_scalar_eltype(∂BB::AbstractArray) =
      isconcretetype(eltype(∂BB)) ? _scalar_eltype(eltype(∂BB)) :
      mapreduce(b -> _scalar_eltype(typeof(b)), promote_type, ∂BB)

function whatalloc(::typeof(pullback!),
                   ∂BB, tensor::SparseACEbasis, Rnl, Ylm
                   )
   # NB: the previous implementation,
   #     TB = eltype.(eltype.(∂BB)); promote_type(..., TB...)
   # splatted a *runtime* `Vector{DataType}` into `promote_type` whenever ∂BB was
   # a `Vector` (e.g. the `pullback([∂B], …)` call in force evaluation). That is
   # not inferrable: `TA` came out as the abstract `DataType`, so the
   # `zeros(TA, …)` in `pullback` produced abstract-eltype arrays and `pullback`
   # was inferred as `Tuple{Any, Any}` (EquivariantTensors.jl#135). Computing the
   # scalar eltype at the type level keeps `TA` a concrete `Type{…}`.
   TB = _blocks_scalar_eltype(∂BB)
   TA = promote_type(eltype(Rnl), eltype(Ylm), TB)
   return (TA, size(Rnl)...), (TA, size(Ylm)...)
end

function pullback(∂BB, tensor::SparseACEbasis{T}, Rnl, Ylm, A) where {T}
   alc_∂Rnl, alc_∂Ylm = whatalloc(pullback!, ∂BB, tensor, Rnl, Ylm)
   ∂Rnl = zeros(alc_∂Rnl...)
   ∂Ylm = zeros(alc_∂Ylm...)
   return pullback!(∂Rnl, ∂Ylm, ∂BB, tensor, Rnl, Ylm, A)
end


# --------------------------------------------------------
#  Row-wise pushforward (Jacobian w.r.t. the per-row inputs)
#
#  Rnl, Ylm : nX x #R, nX x #Y single-node embeddings (one row per neighbour)
#  ∂Rnl, ∂Ylm : one tangent per row, e.g. ∂Rnl[j, n] = ∂Rnl[j, n] / ∂𝐫_j 
#               as an SVector{3}; any tangent type with T * T∂ -> T∂ works 
#  B  : length(tensor) 
#  ∂B : nX x length(tensor), ∂B[j, k] = ∑_t ∂B[k]/∂BB[t][j, :] ⋅ ∂BB[t][j, :]
#
#  This composes the row-wise pushforwards of the A and AA bases with the 
#  coupling map. Only a single scalar (L = 0) output is supported, as for 
#  `_jacobian_X`; for L > 0 the tangents would have to be outer products.

function whatalloc(::typeof(pushforward_rows!), 
                   tensor::SparseACEbasis, Rnl, Ylm, ∂Rnl, ∂Ylm)
   @assert length(tensor.A2Bmaps) == 1 "pushforward_rows! supports a single (L = 0) output"
   A2Bmap = tensor.A2Bmaps[1]
   TA = promote_type(eltype(Rnl), eltype(Ylm))
   TB = _promote_mul_type(eltype(A2Bmap), TA)
   T∂A = _rows_tangent_type(TA, (∂Rnl, ∂Ylm))
   T∂B = _promote_mul_type(TB, T∂A)
   nX = size(Rnl, 1)
   return (TB, length(tensor)), (T∂B, nX, length(tensor))
end

function pushforward_rows!(B::AbstractVector, ∂B::AbstractMatrix, 
                           tensor::SparseACEbasis, 
                           Rnl::AbstractMatrix, Ylm::AbstractMatrix, 
                           ∂Rnl::AbstractMatrix, ∂Ylm::AbstractMatrix)
   @assert length(tensor.A2Bmaps) == 1 "pushforward_rows! supports a single (L = 0) output"
   nX = size(Rnl, 1)
   # one node of the batched _jacobian_X!; ReshapedArray rather than 
   # reshape, which allocates a new Array header (on Julia 1.11) 
   _node3(X) = Base.ReshapedArray(X, (nX, 1, size(X, 2)), ())
   _jacobian_X!(Base.ReshapedArray(B, (1, length(B)), ()), _node3(∂B), tensor, 
                _node3(Rnl), _node3(Ylm), _node3(∂Rnl), _node3(∂Ylm), 
                tensor.abasis.spec, tensor.aabasis.specs, tensor.A2Bmaps[1])
   return B, ∂B
end

function pushforward_rows(tensor::SparseACEbasis, Rnl, Ylm, ∂Rnl, ∂Ylm)
   alc_B, alc_∂B = whatalloc(pushforward_rows!, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)
   B = zeros(alc_B...)
   ∂B = zeros(alc_∂B...)
   return pushforward_rows!(B, ∂B, tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)
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

#   𝔹  : #i x length(tensor) 
#   ∂𝔹 : #j x #i x length(tensor)
#
# This is the batched form of `pushforward_rows!`, which is its one-node 
# case. Only a single (L = 0) output is supported. 

function _jacobian_X(tensor::SparseACEbasis, 
                     Rnl, Ylm, 
                     dRnl, dYlm, 
                     ps, st)
   @assert length(st.A2Bmaps) == 1 "Jacobian currently only supports single basis"
   A2B = st.A2Bmaps[1]
   maxneigs, nnodes, _ = size(Rnl)
   TA = promote_type(eltype(Rnl), eltype(Ylm))
   TB = _promote_mul_type(eltype(A2B), TA)
   T∂B = _promote_mul_type(TB, _rows_tangent_type(TA, (dRnl, dYlm)))
   𝔹 = similar(Rnl, TB, (nnodes, size(A2B, 1)))
   ∂𝔹 = similar(Rnl, T∂B, (maxneigs, nnodes, size(A2B, 1)))
   _jacobian_X!(𝔹, ∂𝔹, tensor, Rnl, Ylm, dRnl, dYlm, 
                st.aspec, st.aaspecs, A2B)
   return (𝔹,), (∂𝔹,)
end

function _jacobian_X!(𝔹::AbstractMatrix, ∂𝔹::AbstractArray{<: Any, 3}, 
                      tensor::SparseACEbasis, 
                      Rnl, Ylm, dRnl, dYlm, 
                      aspec, aaspecs, A2B)
   maxneigs, nnodes, _ = size(Rnl)
   nA, nAA = length(aspec), length(tensor.aabasis)
   TA = promote_type(eltype(Rnl), eltype(Ylm))
   T∂A = _rows_tangent_type(TA, (dRnl, dYlm))
   backend = KernelAbstractions.get_backend(Rnl)
   # on the CPU the intermediates live on the Bumper stack; on a GPU 
   # they are ordinary device arrays 
   @no_escape begin 
      A = _alloc_like(Rnl, TA, nnodes, nA)
      ∂A = _alloc_like(Rnl, T∂A, maxneigs, nnodes, nA)
      _jacobian_X!(A, ∂A, tensor.abasis, aspec, (Rnl, Ylm), (dRnl, dYlm))
      KernelAbstractions.synchronize(backend)
      AA = _alloc_like(Rnl, TA, nnodes, nAA)
      ∂AA = _alloc_like(Rnl, T∂A, maxneigs, nnodes, nAA)
      _jacobian_X!(AA, ∂AA, tensor.aabasis, A, ∂A, aaspecs)
      KernelAbstractions.synchronize(backend)
      # 𝔹 = AA * A2B' and ∂𝔹[:, i, :] = ∂AA[:, i, :] * A2B' 
      _mul_A2Bt!(𝔹, AA, A2B)
      _mul_A2Bt!(∂𝔹, ∂AA, A2B)
      KernelAbstractions.synchronize(backend)
   end
   return nothing 
end

# ::Type{T} so that the element type is inferred (Julia 1.11 does not 
# specialise on a Type argument that is only passed through) 
_alloc_like(X, ::Type{T}, dims...) where {T} = 
      Bumper.alloc!(Bumper.default_buffer(), T, dims...)
_alloc_like(X::AbstractGPUArray, ::Type{T}, dims...) where {T} = 
      similar(X, T, dims)

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

