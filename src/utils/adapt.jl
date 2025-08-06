

import Functors  
using StaticArrays

_floatT(T, A::Any) = A 
_floatT(T, A::AbstractFloat) = T(A) 
_floatT(T, A::AbstractArray{<: AbstractFloat}) = T.(A) 
_floatT(T, A::AbstractArray{<: AbstractArray}) = _float_T.(T, A) 

_float32(A) = _floatT(Float32, A)
_float64(A) = _floatT(Float64, A)

float32(nt::NamedTuple) = Functors.fmap(_float32, nt) 
float64(nt::NamedTuple) = Functors.fmap(_float64, nt)
