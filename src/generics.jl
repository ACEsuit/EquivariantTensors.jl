function pullback end 
function pullback! end 
function pullback2 end 
function pullback2! end 
function pushforward end 
function pushforward! end 


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

pullback2(∂P, ∂X, l::AbstractETLayer, args...) = 
      _with_safe_alloc(pullback2!, ∂P, ∂X, l, args...)
