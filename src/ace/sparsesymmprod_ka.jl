
# interface to use the ka kernel if the output arrays is a GPU array 
function evaluate!(AA::AbstractGPUArray, 
                   basis::SparseSymmProd{ORD},
                   A) where {ORD}
   return ka_evaluate!(AA, basis, A)
end

function ka_evaluate!(AA::AbstractArray, basis::SparseSymmProd{ORD},
                      A::AbstractArray, 
                      specs = basis.specs) where {ORD}
	_ka_evaluate_launcher!(AA, basis, A, specs)
	return AA
end 

function _ka_evaluate_launcher!(AA::AbstractVector,   
                                basis::SparseSymmProd{ORD}, 
                                A::AbstractVector, 
                                specs, ) where {ORD}
   fill!(AA, zero(eltype(AA)))
   @assert length(specs) == ORD   # spec = tuple of sub-specs 
   @assert !(basis.hasconst)     # not implemented
   offsets = first.(basis.ranges)
   
   backend = KernelAbstractions.get_backend(AA)
   kernel! = _ka_evaluate_SparseSymmProd_v1!(backend)

   # @nexprs $ORD N -> 
   for N = 1:ORD 
      kernel!(AA, specs[N], A, Val{N}(), offsets[N]; ndrange = length(specs[N]))
   end
   
   return nothing
end
   
@kernel function _ka_evaluate_SparseSymmProd_v1!(AA, spec, A, ::Val{N}, offset) where {N}
   iAA = @index(Global)
   ϕ = spec[iAA]   # NTuple{N, Int}
   aa = ntuple(t -> A[ϕ[t]], N)  # extract values from A 
   AA[offset+iAA-1] = prod(aa)  # offset by the first element of the range
   nothing 
end

# --------------------------------- 
#  kernels for multiple input 

# on the GPU we assume we always have many inputs at the same time 
# so this is the kernel we should really be using: 
#
# AA = nX x #output-features
#  A = nX x #input-features


function _ka_evaluate_launcher!(AA::AbstractMatrix,   
                                basis::SparseSymmProd{ORD}, 
                                A::AbstractMatrix, 
                                specs, nX = size(A, 1) ) where {ORD}
   fill!(AA, zero(eltype(AA)))
   @assert size(A, 1) >= nX 
   @assert size(AA, 1) >= nX 
   @assert size(AA, 2) >= sum(length, specs) 
   @assert length(specs) == ORD   # spec = tuple of sub-specs 
   @assert !(basis.hasconst)      # not implemented
   offsets = first.(basis.ranges)

   backend = KernelAbstractions.get_backend(AA)
   kernel! = _ka_evaluate_SparseSymmProd_batched_v1!(backend)

   @assert ORD <= 10   # cf. the "10" in @nexprs below 
   @nexprs 10 N -> begin 
      if N <= ORD 
         kernel!(AA, specs[N], A, Val{N}(), offsets[N]; 
                 ndrange = (nX, length(specs[N])))
      end
   end
   
   return nothing
end
   
@kernel function _ka_evaluate_SparseSymmProd_batched_v1!(AA, spec, A, ::Val{N}, offset) where {N}
   iX, iAA = @index(Global, NTuple)
   ϕ = spec[iAA]   # NTuple{N, Int}
   aa = ntuple(t -> A[iX, ϕ[t]], N)  # extract values from A 
   AA[iX, offset+iAA-1] = prod(aa)  # offset by the first element of the range
   nothing 
end
