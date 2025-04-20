
module Test_ACE 



"""
Naive implementation of the product basis, intended only for testing
"""
struct SimpleProdBasis  <: AbstractP4MLTensor
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
                  init = one(eltype(AA)) )
          i = 1:length(basis) ] 
   return AA 
end


end