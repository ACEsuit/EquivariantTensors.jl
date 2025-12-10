
# interface to use the ka kernel if the output arrays is a GPU array 
function evaluate!(AA::AbstractGPUArray, 
                   basis::SparseSymmProd{ORD},
                   A) where {ORD}
   return ka_evaluate!(AA, basis, A)
end

function ka_evaluate(basis::SparseSymmProd{ORD}, A::AbstractArray, 
                     specs = basis.specs) where {ORD}
   AA = similar(A, (size(A, 1), sum(length, specs)))                     
	_ka_evaluate_launcher!(AA, basis, A, specs)
	return AA
end 

function ka_evaluate(basis::SparseSymmProd{ORD}, A::AbstractVector, 
                     specs = basis.specs) where {ORD}
   AA = similar(A, (sum(length, specs),))
	_ka_evaluate_launcher!(AA, basis, A, specs)
	return AA
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

# ---------------------------------
# pullback implementation 

function ka_pullback(∂AA, basis::SparseSymmProd{ORD}, A, 
                     specs = basis.specs, nX = size(∂AA, 1)) where {ORD}
   ∂A = similar(A)
   ka_pullback!(∂A, ∂AA, basis, A, specs, nX)
   return ∂A
end

function ka_pullback!(∂A, ∂AA, basis::SparseSymmProd{ORD}, A, 
                      specs = basis.specs, nX = size(∂AA, 1)) where {ORD}
   @assert size(A, 1) >= nX
   @assert size(∂AA, 1) >= nX
   @assert size(∂A, 1) >= nX 
   @assert size(∂A, 2) == size(A, 2) 
   offsets = first.(basis.ranges)

   backend = KernelAbstractions.get_backend(∂AA)
   kernel! = _ka_pullback_SparseSymmProd_v1!(backend)

   fill!(∂A, zero(eltype(∂A)))

   @assert ORD <= 10   # cf. the "10" in @nexprs below 
   @nexprs 10 N -> begin 
      if N <= ORD 
         kernel!(∂A, ∂AA, A, specs[N], Val{N}(), offsets[N]; 
                 ndrange = (nX, ))
      end
   end
   return ∂A
end

#
# TODO: must replace this with an optimized scatter/gather operation!!!
#
@kernel function _ka_pullback_SparseSymmProd_v1!(∂A, ∂AA, A, 
                                              spec, ::Val{N}, offset) where {N}
   iX = @index(Global)
   for iAA = 1:length(spec)
      ∂AA_cur = ∂AA[iX, offset+iAA-1]
      ϕ = spec[iAA]   # NTuple{N, Int}                                       
      aa = ntuple(t -> A[iX, ϕ[t]], N)  # extract values from A
      _, ∇prod = _static_prod_ed(aa) 
      for j = 1:N 
         ∂A[iX, ϕ[j]] += ∂AA_cur * ∇prod[j]
      end
   end
   nothing 
end
