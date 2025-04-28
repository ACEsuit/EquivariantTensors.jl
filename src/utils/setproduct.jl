
"""
   setproduct(A)

Assumes the `A` is a length-N collection of collections. It returns a 
`P = Matrix{T}` where each `P[i, :]` is a vector of length `N`
with `P[i, j] ∈ A[j]`. The number of columns of `P` is the number of such 
products, i.e. `prod(length.(A))`.

In constrast with `Iterators.product` this implementation 
is type-stable for a priori unknown `N`.
"""
function setproduct(A)
   N = length(A)
   Np = prod(length.(A))

   # guess the element type. 
   p_test = [ A[i][1] for i in 1:N ]
   T = eltype(p_test)

   # allocate storage for the product, add the first element to it and set 
   # a pointer to the P matrix 
   P = Matrix{T}(undef, Np, N)
   idx = 1
   for n = 1:N 
      # corresponds to vv = [1,1...,1] below
      P[idx, n] = A[n][1]   
   end

   # start an index vector at all ones, which represents the multi-index 
   # for the product elements. 
   vv = ones(Int, N) 

   while true
      # figure out which index vv[i] to increment 
      inc = N 
      we_are_done = false 
      while vv[inc] == length(A[inc])
         inc -= 1 
         # if we cannot increment any other index, then we are done 
         if inc == 0 
            we_are_done = true 
            break
         end
      end
      # terminate the outer loop as well if we couldn't find a new vv 
      if we_are_done; break; end

      # increment the current index `inc` and set all previous indices back to 1. 
      vv[inc] += 1
      for i = inc+1:N
         vv[i] = 1
      end

      # now we can write the new product to the matrix
      idx += 1
      for n = 1:N
         P[idx, n] = A[n][vv[n]]
      end

   end 

   # this should be turned on during debugging
   # @assert idx == Np 

   return P 
end