
__TEST_ACE__ = true 

module Test_ACE 

using LinearAlgebra

"""
Naive implementation of the product basis, intended only for testing
"""
struct SimpleProdBasis  
   orders::Vector{Int}
   spec::Matrix{Int} 
end 

Base.length(basis::SimpleProdBasis) = size(basis.spec, 1)


function SimpleProdBasis(specv::AbstractVector{<: AbstractVector}) 
   @assert issorted(length.(specv))
   @assert all(issorted, specv)
   orders = [length(s) for s in specv]
   maxord = maximum(orders)
   specm = zeros(Int, length(specv), maxord)
   for i = 1:length(specv)
      specm[i, 1:orders[i]] = specv[i]
   end
   return SimpleProdBasis(orders, specm)
end 

function (basis::SimpleProdBasis)(A::AbstractVector)
   AA = [ prod( A[basis.spec[i, a]]  for a = 1:basis.orders[i]; 
                  init = one(eltype(A)) )
          for i = 1:length(basis) ] 
   return AA 
end


function generate_SO2_spec(order, M, p=1)
   # m = 0, -1, 1, -2, 2, -3, 3, ... 
   i2m(i) = (-1)^(isodd(i-1)) * (i ÷ 2)
   m2i(m) = 2 * abs(m) - (m < 0)

   spec = Vector{Int}[] 

   function append_N!(::Val{N}) where {N} 
      for ci in CartesianIndices(ntuple(_ -> 1:2*M+1, N))
         mm = i2m.(ci.I)
         if (sum(mm) == 0) && (norm(mm, p) <= M) && issorted(ci.I)
            push!(spec, [ci.I...,])
         end
      end
   end


   for N = 1:order 
      append_N!(Val(N))
   end

   return spec 
end 


end