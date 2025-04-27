

"""
      invmap(a::AbstractVector)

Return a dictionary that maps the elements of a to their indices, to accelerate 
if there are many repeated searches into `a`.
"""
function invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end
