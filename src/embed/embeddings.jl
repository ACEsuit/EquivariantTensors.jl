#
# TODO: likely remove ParallelEmbed
#       likely remove EdgeEmbed 
#       cleanup, polish, improve performance of EdgeEmbed1 
#                 - rename it to EdgeEmbed??
#                 - currently doesn't leverage evaluate_ed on P4ML layer 
#                 - may need switch to DecoratedParticles for that 
#

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


# -------------------------------------------------------------------

# evaluate_ed(emb::ParallelEmbed, X::ETGraph, ps, st) = 
#       _applyparallelembed_ed(emb.layers, X, ps, st)



import ChainRulesCore: rrule, @not_implemented

function rrule(::typeof(evaluate), emb::ParallelEmbed, X::ETGraph, ps, st)
   out, st = evaluate(emb, X, ps, st)

   function _pb_X(∂out) 
      return @not_implemented("backprop w.r.t. X not yet implemented")
   end

   function _pb_ps(∂out) 
      # the simplest case is in fact that there are no parameters, so we 
      # should simply return a ZeroTangent() 
      if sizeof(ps) > 0 
         return @not_implemented("Current implementation of ParallelEmbed rrule does not support parameters!")
      end
      return ZeroTangent() 
   end

   return (out, st), ∂out -> (NoTangent(), NoTangent(), 
                              _pb_X(∂out), _pb_ps(∂out), NoTangent()) 
end



# -------------------------------------------------------------------
#
# alternative, maybe simpler embedding layer that utilizes more 
# of the Lux infrastructure / automatic differentiation.
#


@concrete struct EdgeEmbed <: AbstractLuxWrapperLayer{:layer}
   layer
   name
end

EdgeEmbed(layer; name = "Edge Embedding") = EdgeEmbed(layer, name)

function (l::EdgeEmbed)(X::ETGraph, ps, st)
   Φ2, st = l.layer(X.edge_data, ps, st)
   Φ3 = map(ϕ2 -> reshape_embedding(ϕ2, X), Φ2)
   return Φ3, st
end

#
# This is clever but it should be where reshape_embedding is defined!
# 
function rrule(::typeof(reshape_embedding), ϕ2, X::ETGraph)
   ϕ3 = reshape_embedding(ϕ2, X)

   function _pb_ϕ(∂ϕ3)
      ∂ϕ2 = rev_reshape_embedding(∂ϕ3, X)
      return NoTangent(), ∂ϕ2, NoTangent()
   end

   return ϕ3, _pb_ϕ
end

# function evaluate_ed(l::EdgeEmbed, X::ETGraph, ps, st)

#    Φ2, st = evaluate_ed(l.layer, X.edge_data, ps, st)
#    Φ3 = map(ϕ2 -> reshape_embedding(ϕ2, X), Φ2)

#    ∂ϕ2 = map( DiffNT.grad_fd, 
#    dΦ3 = map( (ϕ2, dϕ2) -> rev_reshape_embedding(dϕ2, X), Φ2, Φ2)
#    return Φ3, dΦ3, st
# end


# -------------------------------------------------------------------
#
# attempt 3: wrapping a single edge embedding (instead of several) 
# into a layer with some additional logic.
# several of these can just be wrapped into a Parallel layer. 
# This seems to provide plenty of logic, and simplifies the code 
#


@concrete struct EdgeEmbed1 <: AbstractLuxWrapperLayer{:layer}
   layer 
   name
end

EdgeEmbed1(layer; name = "Edge Embedding") = 
      EdgeEmbed1(layer, name)

function (l::EdgeEmbed1)(X::ETGraph, ps, st)
   #    #edges x #features
   Φ2, st = l.layer(X.edge_data, ps, st)     
   #    maxneigs x #nodes x #features
   Φ3 = reshape_embedding(Φ2, X)  
   return Φ3, st
end


function evaluate_ed(l::EdgeEmbed1, X::ETGraph, ps, st)
   Φ3, st = l(X, ps, st) 
   ∂Φ2 = mapreduce( x -> permutedims(DiffNT.jac_fd(l.layer, x, ps, st)), 
                    vcat, X.edge_data )
   ∂Φ3 = reshape_embedding(∂Φ2, X)
   return (Φ3, ∂Φ3), st
end
