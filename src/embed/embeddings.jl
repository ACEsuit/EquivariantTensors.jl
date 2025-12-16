#
# TODO: remove ParallelEmbed, and polish EdgeEmbed instead 
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


function rrule(::typeof(reshape_embedding), ϕ2, X::ETGraph)
   ϕ3 = reshape_embedding(ϕ2, X)

   function _pb_ϕ(∂ϕ3)
      ∂ϕ2 = rev_reshape_embedding(∂ϕ3, X)
      return NoTangent(), ∂ϕ2, NoTangent()
   end

   return ϕ3, _pb_ϕ
end


# -------------------------------------------------------------------
#
# Attempt 4: single embedding (can be used inside Parallel)
#   composed of a transform and a basis evaluation 
#   the transform MUST be a mapping from an XState to a valid 
#   input into the basis. 
#   The point of this decomposition is to allow the derivative 
#   through the transform to be computed backward, which is 
#   is very convenient in this setting. 
#   (but could be revisited if we want to optimize performance)
#

using LinearAlgebra: dot 

@concrete struct EdgeEmbedDP <: AbstractLuxContainerLayer{(:trans, :basis)}
   trans
   basis
   name
end

EdgeEmbedDP(trans, basis; name = "") = 
         EdgeEmbedDP(trans, basis, name)

Base.show(io::IO, ::MIME"text/plain", l::EdgeEmbedDP) = 
      print(io, "EdgeEmbedDP($(l.name))")         


function (l::EdgeEmbedDP)(X::ETGraph, ps, st)
   # first gets rid of the state variable in the return 
   Y = broadcast(first ∘ l.trans, X.edge_data, Ref(ps.trans), Ref(st.trans))
   Φ2, _ = l.basis(Y, ps.basis, st.basis)
   Φ3 = reshape_embedding(Φ2, X)
   return Φ3, st
end

function evaluate_ed(l::EdgeEmbedDP, X::ETGraph, ps, st)
   # first gets rid of the state variable in the return 
   ftrans = _x -> first(l.trans(_x, ps.trans, st.trans))
   Y = broadcast(ftrans, X.edge_data)
   Φ2, dΦ2 = evaluate_ed(l.basis, Y, ps.basis, st.basis)

   # pullback through the transform to get ∂Φ2
   _pb1(x, dφ) = DiffNT.grad_fd(_x -> dot(ftrans(_x), dφ), x)
   ∂Φ2 = broadcast(_pb1, X.edge_data, dΦ2)
   
   Φ3 = reshape_embedding(Φ2, X)
   ∂Φ3 = reshape_embedding(∂Φ2, X)
   return (Φ3, ∂Φ3), st
end



# function rrule(::typeof(reshape_embedding), ϕ2, X::ETGraph)
#    ϕ3 = reshape_embedding(ϕ2, X)

#    function _pb_ϕ(∂ϕ3)
#       ∂ϕ2 = rev_reshape_embedding(∂ϕ3, X)
#       return NoTangent(), ∂ϕ2, NoTangent()
#    end

#    return ϕ3, _pb_ϕ
# end

