
struct FFInvMap{T} 
   a::Vector{T} 
   p::Vector{Int} 
end 

Base.getindex(m::FFInvMap{T}, v::T) where {T} = m.p[searchsortedfirst(m.a, v)]

"""
      invmap(a::AbstractVector)

Returns a structure that makes looking up the index of an element in a vector 
convenient and fast. Assumes that elements of `a` are unique.
```julia
inva = invmap(a) 
inva[a[i]] == i  # true for all i
```
"""
function invmap(a::AbstractVector)
   p = sortperm(a) 
   b = a[p]
   if !allunique(b)
      throw(ArgumentError("invmap: Elements of a must be unique"))
   end
   return FFInvMap(b, p)
end


# Backup function which is probably not needed anymore. 
function dict_invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end
