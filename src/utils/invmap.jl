
struct FFInvMap{T, HF} 
   h::Vector{T} 
   p::Vector{Int} 
   hashfcn::HF
end 

function Base.getindex(m::FFInvMap{T}, v) where {T}
   h = m.hashfcn(v)::T
   i = searchsortedfirst(m.h, h)
   return m.p[i]
end

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
   h = hashfcn.(a)
   p = sortperm(h) 
   permute!(h, p)
   if any(h[i] == h[i+1] for i = 1:length(h)-1)
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
