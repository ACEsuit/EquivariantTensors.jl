
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
