module O3

using StaticArrays
 
import EquivariantTensors.SO3 as SO3
import EquivariantTensors.SO3: mm_filter

export coupling_coeffs

include("O3_utils.jl")
include("yyvector.jl")
include("O3_transformations.jl")

coupling_coeffs(L::Integer, ll, nn = nothing; 
                         PI = !(isnothing(nn)), 
                         basis = complex) = 
    iseven(sum(ll)+L) ? SO3.coupling_coeffs(L, ll, nn; PI = PI, basis = basis) : 
                        (zeros(Float64, 0, 0), SVector{length(ll), Int}[])

end