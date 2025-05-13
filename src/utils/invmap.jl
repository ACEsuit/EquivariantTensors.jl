
struct FFInvMap{T, HF} 
   h::Vector{T} 
   p::Vector{Int} 
   hashfcn::HF
end 

Base.getindex(m::FFInvMap, v) = 
      m.p[searchsortedfirst(m.h, m.hashfcn(v))]

"""
      invmap(a::AbstractVector)

Returns a structure that makes looking up the index of an element in a vector 
convenient and fast. Assumes that elements of `a` are unique.
```julia
inva = invmap(a) 
inva[a[i]] == i  # true for all i
```
"""
function invmap(a::AbstractVector, hashfcn = identity)
   p = sortperm(a) 
   h = [ hashfcn(a[p[i]]) for i in 1:length(a) ]
   if !allunique(h)
      throw(ArgumentError("invmap: Elements of a must be unique"))
   end
   return FFInvMap(h, p, hashfcn)
end


# Backup function which is probably not needed anymore. 
function dict_invmap(a::AbstractVector)
   inva = Dict{eltype(a), Int}()
   for i = 1:length(a) 
      inva[a[i]] = i 
   end
   return inva 
end
