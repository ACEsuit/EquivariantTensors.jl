
_promote_mul_type(args...) = promote_type(args...)

_promote_mul_type(TA::Type{<: Number}, TB::Type{SVector{N, TC}}
                  ) where {N, TC} =
         SVector{N, promote_type(TA, TC)}
