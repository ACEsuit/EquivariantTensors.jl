#
# Scatter/gather implementation of PooledSparseProduct
#
# The fused operation  A[iA] = Σⱼ Πₜ BB[t][j, spec[iA][t]]
# is decomposed into:
#   Forward:  gather → elementwise product → sum
#   Backward: broadcast → product rule → scatter-add
#
# This avoids write conflicts in the backward pass (the fused
# kernel has += to ∂BB[t] from multiple iA with the same column
# index). The scatter-add collects those contributions cleanly.
#

using KernelAbstractions
using KernelAbstractions: @atomic
using Adapt: adapt
using EquivariantTensors: PooledSparseProduct, TupTen3

# ============================================================
#  Index precomputation
# ============================================================

"""
    GatherIndices{NB}

Precomputed index arrays for the gather/scatter prodpool.
`idx[t]` is a `Vector{Int}` (or GPU array) of length `nA` mapping
each spec element to its column in `BB[t]`.
"""
struct GatherIndices{NB, TI}
   idx::NTuple{NB, TI}
   nA::Int
end

function GatherIndices(
      basis::PooledSparseProduct{NB}, BB,
      ) where {NB}
   spec = basis.spec
   nA = length(spec)
   idx_cpu = ntuple(
      t -> [spec[iA][t] for iA in 1:nA], NB)
   # move to the same backend as BB
   backend = KernelAbstractions.get_backend(BB[1])
   if backend == KernelAbstractions.CPU()
      idx = idx_cpu
   else
      idx = ntuple(
         t -> adapt(backend, idx_cpu[t]), NB)
   end
   return GatherIndices{NB, typeof(idx[1])}(idx, nA)
end


# ============================================================
#  Forward: gather-prod-pool
# ============================================================

"""
    sg_evaluate(basis, BB, [gidx]; nneig, nnodes)

Scatter/gather evaluate for the batched (TupTen3) case.

BB[t] is (nneig × nnodes × nfeat_t).
Output A is (nnodes × nA).
"""
function sg_evaluate(
      basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   nA = gidx.nA
   A = similar(BB[1], (nnodes, nA))
   sg_evaluate!(A, basis, BB; gidx, nneig, nnodes)
   return A
end

function sg_evaluate!(
      A::AbstractMatrix,
      basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   nA = gidx.nA
   @assert size(A, 1) >= nnodes
   @assert size(A, 2) >= nA

   # Step 1: Gather columns → G[t] is (nneig × nnodes × nA)
   G = ntuple(
      t -> _gather3(BB[t], gidx.idx[t]), NB)

   # Step 2+3: elementwise product and sum over nneig
   #   A[inode, iA] = Σ_ineig Π_t G[t][ineig, inode, iA]
   backend = KernelAbstractions.get_backend(A)
   fill!(A, zero(eltype(A)))
   kernel! = _ka_prod_pool!(backend)
   kernel!(A, G, Val{NB}(); ndrange = (nneig, nnodes, nA))

   return A
end

# Gather along dim 3: out[:, :, iA] = src[:, :, idx[iA]]
function _gather3(src::AbstractArray{T, 3}, idx) where {T}
   nneig, nnodes, _ = size(src)
   nA = length(idx)
   out = similar(src, (nneig, nnodes, nA))
   backend = KernelAbstractions.get_backend(src)
   kernel! = _ka_gather3!(backend)
   kernel!(out, src, idx; ndrange = (nneig, nnodes, nA))
   return out
end

@kernel function _ka_gather3!(out, src, idx)
   i, j, k = @index(Global, NTuple)
   @inbounds out[i, j, k] = src[i, j, idx[k]]
end

@kernel function _ka_prod_pool!(
      A, G, ::Val{NB}) where {NB}
   ineig, inode, iA = @index(Global, NTuple)
   @inbounds begin
      p = G[1][ineig, inode, iA]
      for t = 2:NB
         p *= G[t][ineig, inode, iA]
      end
      @atomic A[inode, iA] += p
   end
end


# ============================================================
#  Backward: scatter-add pullback
# ============================================================

"""
    sg_pullback(∂A, basis, BB; gidx, ...)

Returns ∂BB, a tuple of arrays with same shapes as BB.
"""
function sg_pullback(
      ∂A, basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   ∂BB = ntuple(t -> similar(BB[t]), NB)
   sg_pullback!(∂BB, ∂A, basis, BB;
                gidx, nneig, nnodes)
   return ∂BB
end

function sg_pullback!(
      ∂BB, ∂A,
      basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   nA = gidx.nA

   # Step 1: Gather (same as forward)
   G = ntuple(
      t -> _gather3(BB[t], gidx.idx[t]), NB)

   # Step 2: Compute ∂G[t] = ∂A[inode, iA] * Π_{s≠t} G[s]
   #         (nneig × nnodes × nA) for each t
   ∂G = ntuple(
      t -> similar(BB[t], (nneig, nnodes, nA)), NB)

   backend = KernelAbstractions.get_backend(∂A)
   kernel! = _ka_grad_prod!(backend)
   kernel!(∂G, ∂A, G, Val{NB}();
           ndrange = (nneig, nnodes, nA))

   # Step 3: Scatter-add ∂G[t] back into ∂BB[t]
   for t = 1:NB
      fill!(∂BB[t], zero(eltype(∂BB[t])))
      _scatter_add3!(∂BB[t], ∂G[t], gidx.idx[t])
   end

   return ∂BB
end

@kernel function _ka_grad_prod!(
      ∂G, ∂A, G, ::Val{NB}) where {NB}
   ineig, inode, iA = @index(Global, NTuple)
   @inbounds begin
      ∂A_val = ∂A[inode, iA]
      # compute all b values
      for t = 1:NB
         # product of all factors except t
         p = one(eltype(∂A))
         for s = 1:NB
            if s != t
               p *= G[s][ineig, inode, iA]
            end
         end
         ∂G[t][ineig, inode, iA] = ∂A_val * p
      end
   end
end

# Scatter-add along dim 3: dst[:, :, idx[k]] += src[:, :, k]
function _scatter_add3!(
      dst::AbstractArray{T, 3},
      src::AbstractArray{T, 3},
      idx) where {T}
   nneig, nnodes, nA = size(src)
   backend = KernelAbstractions.get_backend(dst)
   kernel! = _ka_scatter_add3!(backend)
   kernel!(dst, src, idx;
           ndrange = (nneig, nnodes, nA))
   return dst
end

@kernel function _ka_scatter_add3!(dst, src, idx)
   i, j, k = @index(Global, NTuple)
   @inbounds @atomic dst[i, j, idx[k]] += src[i, j, k]
end


# ============================================================
#  Fused-scatter: no intermediate arrays
# ============================================================
#
# Same (ineig, inode, iA) parallelization as scatter/gather
# but reads BB directly via idx — no materialized G or ∂G.
# Forward: keeps the existing fused kernel (no atomics needed).
# Backward: one atomic add per thread per factor.

using EquivariantTensors: _static_prod_ed

"""
    fs_pullback(∂A, basis, BB; gidx, ...)

Fused-scatter pullback: same parallelization as scatter/gather
but without materializing intermediate arrays.
"""
function fs_pullback(
      ∂A, basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   ∂BB = ntuple(t -> similar(BB[t]), NB)
   fs_pullback!(∂BB, ∂A, basis, BB;
                gidx, nneig, nnodes)
   return ∂BB
end

function fs_pullback!(
      ∂BB, ∂A,
      basis::PooledSparseProduct{NB}, BB::TupTen3;
      gidx = GatherIndices(basis, BB),
      nneig = size(BB[1], 1),
      nnodes = size(BB[1], 2),
      ) where {NB}
   nA = gidx.nA
   for t = 1:NB
      fill!(∂BB[t], zero(eltype(∂BB[t])))
   end
   backend = KernelAbstractions.get_backend(∂A)
   kernel! = _ka_pullback_fused_scatter!(backend)
   kernel!(∂BB, ∂A, BB, gidx.idx, Val{NB}();
           ndrange = (nneig, nnodes, nA))
   return ∂BB
end

@kernel function _ka_pullback_fused_scatter!(
      ∂BB, ∂A, BB, idx, ::Val{NB}) where {NB}
   ineig, inode, iA = @index(Global, NTuple)
   @inbounds begin
      ∂A_val = ∂A[inode, iA]
      # read factors directly from BB via idx
      b = ntuple(t -> BB[t][ineig, inode, idx[t][iA]], NB)
      _, g = _static_prod_ed(b)
      for t = 1:NB
         @atomic ∂BB[t][ineig, inode, idx[t][iA]] +=
            ∂A_val * g[t]
      end
   end
end
