function pullback end 
function pullback! end 
function pullback2 end 
function pullback2! end 
function pushforward end 
function pushforward! end 

"""
   pushforward_rows(layer, X, ∂X) -> (Y, ∂Y)
   pushforward_rows!(Y, ∂Y, layer, X, ∂X) -> (Y, ∂Y)

Row-wise pushforward: `X` is a single (possibly pooled) input whose rows 
are the individual inputs (e.g. neighbours, edges) and `∂X[j, :]` is one 
tangent per row `j`. `∂Y[j, :]` is the pushforward of `∂X[j, :]` alone; 
nothing is summed over `j`. The tangent type only needs to satisfy 
`T * T∂ -> T∂`, so with `∂X[j, k] = ∂X[j, k] / ∂𝐫_j :: SVector{3}` the 
output is the Jacobian `∂Y[j, k] = ∂Y[k] / ∂𝐫_j`. 

Compare with `pushforward!`, which propagates a single tangent (of the 
same shape as the input) and sums over the pooled rows. 
"""
function pushforward_rows end 
function pushforward_rows! end 


# a helper that converts all whatalloc outputs to tuple form 
function _tup_whatalloc(args...) 
   _to_tuple(wa::Tuple{Vararg{Tuple}}) = wa 
   _to_tuple(wa::Tuple{<: Type, Vararg{Integer}}) = (wa,)
   return _to_tuple(whatalloc(args...))
end

# _with_safe_alloc is a simple analogy of WithAlloc.@withalloc 
# that allocates standard arrays on the heap instead of using Bumper 
function _with_safe_alloc(fcall, args...) 
   allocinfo = _tup_whatalloc(fcall, args...)
   outputs = ntuple(i -> zeros(allocinfo[i]...), length(allocinfo))
   return fcall(outputs..., args...)
end

(l::AbstractETLayer)(args...) = 
      evaluate(l, args...)
            
evaluate(l::AbstractETLayer, args...) = 
      _with_safe_alloc(evaluate!, l, args...) 

pullback(∂X, l::AbstractETLayer, args...) = 
      _with_safe_alloc(pullback!, ∂X, l, args...)

pushforward(l::AbstractETLayer, args...) = 
      _with_safe_alloc(pushforward!, l, args...)

pushforward_rows(l::AbstractETLayer, args...) = 
      _with_safe_alloc(pushforward_rows!, l, args...)

pullback2(∂P, ∂X, l::AbstractETLayer, args...) = 
      _with_safe_alloc(pullback2!, ∂P, ∂X, l, args...)
