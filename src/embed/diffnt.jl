#
# TODO: get rid of NT and replace with DP 
#

module DiffNT 

using ForwardDiff, NamedTupleTools, StaticArrays
using ForwardDiff: Dual 

_iscts(::Any) = false 
_iscts(::Type{T}) where {T <: AbstractFloat} = true 
_iscts(::Type{Dual{T1, T2, T3}}) where {T1, T2, T3} = _iscts(T2) 
_iscts(T::Type{<: SArray}) = _iscts(eltype(T))


"""
   _ctsnt(x::NamedTuple) 

From a `nt::NamedTuple` extract only the continuous variables and return 
a new NamedTuple with those variables.
"""
@generated function _ctsnt(x::NamedTuple)
   SYMS = fieldnames(x)
   TT = fieldtypes(x)

   code = "(; "
   for (sym, T) in zip(SYMS, TT)
      if _iscts(T)
         code *= "$sym = x.$sym,"
      end
   end
   code *= ")"
   return quote 
      $(Meta.parse(code))
   end 
end

"""
   _replace(x::NamedTuple, v::NamedTuple)

Assuming that `fieldnames(v)` is a subset of `fieldnames(x)`, this constructs 
a new `NamedTuple` `y` with the same fields as `x`, but with the values of `v` replacing
the values of `x` for the fields that are present in `v`.   
"""
@generated function _replace(x::NamedTuple, v::NamedTuple)
   SYMSX = fieldnames(x)
   SYMSV = fieldnames(v)

   code = "(; "
   for sym in SYMSX
      if sym in SYMSV
         code *= "$sym = v.$sym,"
      else 
         code *= "$sym = x.$sym,"
      end
   end 
   code *= ")"
   return quote 
      $(Meta.parse(code))
   end 
end

"""
   svector_type(v::NamedTuple) 

Assuming that all fields of `v` are either of type `T` or of type `SVector{N, T}`,
this generates a type `SVector{N, T}` where `N` is the total number of elements
and `T` is the unique type of the elements. If there are multiple types, it raises an error.   
"""
@generated function svector_type(v::NamedTuple) 
   TT = fieldtypes(v)

   _tfl(::Any) = Any 
   _tfl(::Type{T}) where {T <: Number} = T
   _tfl(::Type{SVector{N, T}}) where {N, T <: Number} = T 
   T = unique(_tfl(T) for T in TT)
   
   if length(T) > 1 
      ex = :( error("only one float type allowed") )
   else 

      _len(::Type{T}) where {T <: AbstractFloat} = 1
      _len(::Type{SVector{N, T}}) where {N, T <: AbstractFloat} = N

      T = first(T)  # the only type
      len = sum(_len, TT) 
      ex = Meta.parse("SVector{$len, $T}")
   end 

   return quote 
      $(ex)
   end
end

"""
   _nt2svec(x::NamedTuple)

Converts a `NamedTuple` `x` into a `SVector` of the type generated 
by `svector_type(x)`, i.e. flattening the data stored in `x` into a single vector.
"""
_nt2svec(x::NamedTuple) = reinterpret(svector_type(x), x) 

"""
   _svec2nt(v::SVector, x::NamedTuple)

Converts a `SVector` `v` into a `NamedTuple` with the same fields as `x`,
where each field is filled with the corresponding elements from `v`. The 
NamedTuple `x` acts as a "blueprint" for the structure of the output NamedTuple, 
specifying the field names and the length of each field. The SVector `v` 
provides the data and the data type. 
"""
@generated function _svec2nt(v::SVector, x::NamedTuple) 
   SYMS = fieldnames(x)
   TT = fieldtypes(x)

   _len(::Type{T}) where {T <: Number} = 1
   _len(::Type{SVector{N, T}}) where {N, T <: Number} = N

   i0 = Int[] 
   idx = 1 
   for T in TT
      push!(i0, idx) 
      idx += _len(T)
   end
   push!(i0, idx) 

   # indexing into v::SVector 
   inds = [] 
   for i = 1:length(TT) 
      rg = i0[i]:i0[i+1]-1
      if length(rg) == 1 
         push!(inds, "$(first(rg))")
      else 
         rg = SVector(rg...)
         push!(inds, "SA$rg")
      end
   end

   code = "(; "
   for (sym, ind) in zip(SYMS, inds)
      code *= "$sym = v[$ind], "
   end 
   code *= ")"

   return quote 
      $(Meta.parse(code))
   end
end 


"""
   grad_fd(f, x::NamedTuple, args...)

`ForwardDiff` gradient of a function `f` with respect to the continuous 
variables stored in the NamedTuple `x`; returns a NamedTuple with 
the gradient values corresponding to the continuous variables in `x`. 
The `args...` are taken as constant paramteters during this differentiation. 
"""
function grad_fd(f, x::NamedTuple, args...)
   v_nt = _ctsnt(x)  # extract continuous variables into an SVector 
   _fvec = _v -> f(_replace(x, _svec2nt(_v, v_nt)), args...)
   g = ForwardDiff.gradient(_fvec, _nt2svec(v_nt))
   return _svec2nt(g, v_nt)  # return as NamedTuple
end 


end 