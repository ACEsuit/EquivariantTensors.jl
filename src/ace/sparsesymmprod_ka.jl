
# interface to use the ka kernel if the output arrays is a GPU array 
function evaluate!(AA::AbstractGPUArray, 
                   basis::SparseSymmProd{ORD},
                   A) where {ORD}
   return ka_evaluate!(AA, basis, A)
end

function ka_evaluate!(AA::AbstractVector, basis::SparseSymmProd{ORD},
                      A::AbstractVector, 
                      specs = basis.specs) where {ORD}
	_ka_evaluate_launcher!(AA, basis, A, specs)
	return AA
end 

function _ka_evaluate_launcher!(AA::AbstractVector,   
                                basis::SparseSymmProd{ORD}, 
                                A::AbstractVector, specs) where {ORD}
   fill!(AA, zero(eltype(AA)))
   @assert length(specs) == ORD   # spec = tuple of sub-specs 
   @assert !(basis.hasconst)     # not implemented
   # offsets = first(basis.ranges[N])
   
   backend = KernelAbstractions.get_backend(AA)
   kernel! = _ka_evaluate_SparseSymmProd_v1!(backend)

   # @nexprs @ORD N -> 
   for N = 1:ORD 
      offset = first(basis.ranges[N])
      spec_N = specs[N]  # NTuple{N, Int}
      kernel!(AA, spec_N, A, Val{N}(), offset; ndrange = length(spec_N))
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

# on the GPU we assume we always have many inputs at the same time, hence TupMat 
