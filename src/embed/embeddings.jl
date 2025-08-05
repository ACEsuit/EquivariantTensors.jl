using ConcreteStructs 
using Lux: AbstractLuxWrapperLayer
import Lux 

"""
`struct ParallelEmbed` : basically a variation on Lux.Parallel, 
but makes some assumptions on what the individual layers to 
be evaluated in parallel do to enable some simple optimizations. 

The assumption is that each layer is a "basis", i.e. a mapping x -> B(x), 
where x is low-dimensional and B(x) is moderate-dimensional. This means that 
the optimal differentiation is in forward-mode. This is exploited by 
implementing the functions `evaluate` and `evaluate_ed`. For implementing 
models that use `ParallelEmbed` a forward-pass that also differentiates 
(e.g. energy and forces) is implemented with `evaluate_ed`. 
"""
@concrete struct ParallelEmbed <: AbstractLuxWrapperLayer{:layers}
   layers <: NamedTuple
   name
end

ParallelEmbed(layers...; name = "Embedding") = 
      ParallelEmbed((;layers...), name)

ParallelEmbed(; name="Embedding", kwargs...) = 
      ParallelEmbed((; kwargs...), name)


evaluate(emb::ParallelEmbed, X::ETGraph, ps, st) = 
      _applyparallelembed(emb.layers, X, ps, st)

(emb::ParallelEmbed)(args...) = evaluate(emb, args...)      

# This here is almost copy-pasted from Lux.jl; with very minor difference, 
# The real assumptions are made once we differentiate. 
#
@generated function _applyparallelembed(layers::NamedTuple{names}, 
                                 X::ETGraph, ps, st::NamedTuple) where {names}
   N = length(names)
   y_symbols = ntuple(i -> gensym(), N)
   y3_symbols = ntuple(i -> gensym(), N)
   st_symbols = ntuple(i -> gensym(), N)
   calls = []
   append!(calls,
         [ :(
                ($(y_symbols[i]), $(st_symbols[i])) = evaluate(
                    layers.$(names[i]), X.edge_data, ps.$(names[i]), st.$(names[i]) )
            ) for i in 1:N ], )
   append!(calls,
         [ :( $(y3_symbols[i]) = reshape_embedding($(y_symbols[i]), X) ) 
           for i in 1:N ], )
   push!(calls, :(out = ($(y3_symbols...),)))
   push!(calls, :(st = NamedTuple{$names}((($(st_symbols...),)))))
   push!(calls, :(return out, st))
   return Expr(:block, calls...)
end

import ChainRulesCore: rrule 
function rrule(::typeof(evaluate), emb::ParallelEmbed, X::ETGraph, ps, st)
   out, st = evaluate(emb, X, ps, st)

   function _pb_ps(∂out) 
      @show "blurg"
      error("stop")
   end

   return (out, st), ∂out -> (NoTangent(), NoTangent(), NoTangent(), 
                              _pb_ps(∂out), NoTangent()) 
end
