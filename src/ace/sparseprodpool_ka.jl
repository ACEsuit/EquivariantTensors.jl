
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

# on the GPU we assume we always have many inputs at the same time, hence TupMat 
