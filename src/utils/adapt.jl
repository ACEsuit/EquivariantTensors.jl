

import Functors  
using StaticArrays
using DecoratedParticles: XState

_floatT(T, A::Any) = A 
_floatT(T, A::AbstractFloat) = T(A) 
_floatT(T, A::AbstractArray{<: AbstractFloat}) = T.(A) 
_floatT(T, A::AbstractArray{<: AbstractArray}) = _float_T.(T, A) 

_float32(A) = _floatT(Float32, A)
_float64(A) = _floatT(Float64, A)

float32(x::AbstractFloat) = Float32(x)
float64(x::AbstractFloat) = Float64(x)

float32(nt::NamedTuple) = Functors.fmap(_float32, nt) 
float64(nt::NamedTuple) = Functors.fmap(_float64, nt)

float32(x::T) where {T <: XState} = T( float32(getfield(x, :x)) )
float64(x::T) where {T <: XState} = T( float64(getfield(x, :x)) )

float32(::Nothing) = nothing 

float32(X::ETGraph) = 
   ETGraph( X.ii, X.jj, X.first, 
            float32.(X.node_data), 
            float32.(X.edge_data), 
            float32.(X.graph_data),
            X.maxneigs )
