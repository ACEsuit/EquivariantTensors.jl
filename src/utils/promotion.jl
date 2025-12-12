
import ForwardDiff: Dual

_promote_mul_type(args...) = promote_type(args...)

_promote_mul_type(TA::Type{<: Number}, TB::Type{SVector{N, TC}}
                  ) where {N, TC} =
         SVector{N, promote_type(TA, TC)}


# ============================================================================
#   Dual-aware type promotion for ForwardDiff compatibility
# ============================================================================

"""
    _extract_base_type(T)

Extract the base numeric type from a (possibly nested) ForwardDiff.Dual type.
"""
_extract_base_type(::Type{T}) where {T <: Real} = T
_extract_base_type(::Type{Dual{Tag, V, N}}) where {Tag, V, N} = _extract_base_type(V)

"""
    _is_dual_type(T)

Check if type T is a ForwardDiff.Dual type.
"""
_is_dual_type(::Type{<:Real}) = false
_is_dual_type(::Type{<:Dual}) = true

"""
    _promote_type_dual(types...)

Type promotion that preserves ForwardDiff.Dual wrappers.

Standard `promote_type` can lose Dual information when mixing Dual and non-Dual
types. This function extracts the base types, promotes them, and rewraps in Dual
if any input was a Dual type.

# Examples
```julia
_promote_type_dual(Float64, Float32)  # Returns Float64
_promote_type_dual(Dual{Tag,Float64,N}, Float32)  # Returns Dual{Tag,Float64,N}
_promote_type_dual(Dual{Tag,Float32,N}, Float64)  # Returns Dual{Tag,Float64,N}
```
"""
function _promote_type_dual(types...)
    # Extract base types from any Duals
    base_types = map(_extract_base_type, types)
    promoted_base = promote_type(base_types...)

    # If any input was Dual, return the first Dual type with promoted base
    for T in types
        if _is_dual_type(T)
            # Reconstruct Dual with the promoted base type
            Tag = T.parameters[1]
            N = T.parameters[3]
            return Dual{Tag, promoted_base, N}
        end
    end

    return promoted_base
end

# Convenience: promote element types of arrays while preserving Dual
_promote_eltype_dual(arrays...) = _promote_type_dual(map(eltype, arrays)...)
