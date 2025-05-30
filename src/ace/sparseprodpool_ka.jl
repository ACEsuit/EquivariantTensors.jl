
# interface to use the ka kernel if the output arrays is a GPU array 
function evaluate!(A::AbstractGPUArray, 
                   basis::PooledSparseProduct{NB}, 
                   BB::TupVec) where {NB}
   return ka_evaluate!(A, basis, BB)
end


function ka_evaluate!(A::AbstractVector, basis::PooledSparseProduct{NB}, 
                      BB::TupMat, 
                      spec = basis.spec, 
                      nX = size(BB[1],1)) where {NB}
	_ka_evaluate_launcher!(A, basis, BB, spec, nX)
	return A
end 

function _ka_evaluate_launcher!(A, basis::PooledSparseProduct{NB}, BB::TupMat, 
                                spec, nX) where {NB}
   @assert all(B->size(B, 1) >= nX, BB)
   fill!(A, zero(eltype(A)))
   backend = KernelAbstractions.get_backend(A)
   kernel! = _ka_evaluate_PooledSparseProduct_v1!(backend)
   kernel!(A, BB, spec, nX, Val{NB}(); ndrange = (length(spec), nX))
   return nothing
end
   
@kernel function _ka_evaluate_PooledSparseProduct_v1!(A, BB, spec, nX, ::Val{NB}) where {NB}
   iA, j = @index(Global, NTuple)
   ϕ = spec[iA]
   b = ntuple(t -> BB[t][j, ϕ[t]], NB)

   @atomic A[iA] += prod(b)
end

# --------------------------------- 
#  kernels for multiple input 

# on the GPU we assume we always have many inputs at the same time
#
#      A = #nodes x #output-features 
#  BB[t] = #neighbours x #nodes x #input-features[t]

function ka_evaluate(basis::PooledSparseProduct{NB}, 
                     BB::TupTen3, 
                     spec = basis.spec, 
                     nX = size(BB[1], 2), nneig = size(BB[1], 1)
                     ) where {NB}
   A = similar(BB[1], (nX, length(spec)))                     
   return ka_evaluate!(A, basis, BB, spec, nX, nneig)
end


function ka_evaluate!(A::AbstractMatrix, basis::PooledSparseProduct{NB}, 
                      BB::TupTen3, 
                      spec = basis.spec, 
                      nX = size(BB[1], 2), nneig = size(BB[1], 1)
                      ) where {NB}
	_ka_evaluate_launcher!(A, basis, BB, spec, nX, nneig)
	return A
end 

function _ka_evaluate_launcher!(
                     A::AbstractMatrix, basis::PooledSparseProduct{NB}, 
                     BB::TupTen3, spec, nX, nneig 
                     ) where {NB}
   # check correct number of nodes                      
   @assert all(B -> size(B, 2) >= nX, BB)
   @assert size(A, 1) >= nX 
   # check correct number of neighbours 
   @assert all(B -> size(B, 1) >= nneig, BB)

   fill!(A, zero(eltype(A)))
   backend = KernelAbstractions.get_backend(A)
   kernel! = _ka_evaluate_PooledSparseProduct_batched_v1!(backend)
   kernel!(A, BB, spec, Val{NB}(); ndrange = (length(spec), nX, nneig))
   return nothing
end
   
@kernel function _ka_evaluate_PooledSparseProduct_batched_v1!(A, BB, spec, ::Val{NB}) where {NB}
   iA, inode, ineig = @index(Global, NTuple)
   ϕ = spec[iA]
   b = ntuple(t -> BB[t][ineig, inode, ϕ[t]], NB)
   a = prod(b) 
   @atomic A[inode, iA] += a
end

# ---------------------------------
#  pullback
function ka_pullback(∂A, basis::PooledSparseProduct{NB}, 
                      BB::TupTen3, 
                      spec = basis.spec, nX = size(∂A, 1), nneig = size(BB[1], 1)
                      ) where {NB}
   ∂BB = similar.(BB)
   ka_pullback!(∂BB, ∂A, basis, BB, spec, nX, nneig)
   return ∂BB
end

function ka_pullback!(∂BB, ∂A, basis::PooledSparseProduct{NB}, BB::TupTen3, 
                      spec = basis.spec, nX = size(∂A, 1), 
                      nneig = size(BB[1], 1)) where {NB}

   @assert all(B -> size(B, 2) >= nX, BB)
   @assert all(B -> size(B, 1) >= nneig, BB)
   @assert size(∂A, 1) >= nX 
   @assert size(∂A, 2) >= length(spec)

   for t = 1:NB 
      fill!(∂BB[t], zero(eltype(∂BB[t])))
   end

   backend = KernelAbstractions.get_backend(∂A)
   kernel! = _ka_pullback_PooledSparseProduct_v1!(backend)
   kernel!(∂BB, ∂A, BB, spec, nX, nneig, Val{NB}();
           ndrange = (length(spec), nX, nneig))
   return nothing
end


@kernel function _ka_pullback_PooledSparseProduct_v1!(
                  ∂BB, ∂A, BB, spec, nX, nneig, ::Val{NB}) where {NB}
   iA, inode, ineig = @index(Global, NTuple) 
   ϕ = spec[iA]
   b = ntuple(t -> BB[t][ineig, inode, ϕ[t]], NB)
   p, ∇prod = _static_prod_ed(b)
   # A[inode, iA] += p
   for t = 1:NB
      @atomic ∂BB[t][ineig, inode, ϕ[t]] += ∂A[inode, iA] * ∇prod[t]
   end
   nothing 
end 

   # @inbounds for (iA, ϕ) in enumerate(basis.spec)
   #    ∂A_iA = ∂A[iA]
   #    @simd ivdep for j = 1:nX 
   #       b = ntuple(Val(NB)) do i 
   #          BB[i][j, ϕ[i]] 
   #       end 
   #       a, g = _static_prod_ed(b)
   #       for i = 1:NB 
   #          ϕi = ϕ[i]
   #          ∂BB[i][j, ϕi] = muladd(∂A_iA, g[i], ∂BB[i][j, ϕi])
   #       end
   #    end 
   # end
