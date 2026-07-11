
# EdgeEmbed: the adapter between a per-edge embedding layer and an `ETGraph`.
# It applies the wrapped layer to the graph's `edge_data` and `reshape_embedding`s
# the result into the graph-aware 3-tensor (and provides `evaluate_ed` for
# jacobians). Lives in `graphs/` because it is primarily a graph operation — the
# reshape machinery (`reshape_embedding` / `rev_reshape_embedding`) is defined in
# `graph.jl`.

using ConcreteStructs
using LuxCore: AbstractLuxWrapperLayer


"""
   struct EdgeEmbed

Wraps a layer that embeds an edge state into a vector, managing the reformatting
of the embedding from a list of embedded states into a graph-aware 3-tensor.
Also implements an `evaluate_ed` wrapper, useful for computing jacobians.
"""
@concrete struct EdgeEmbed <: AbstractLuxWrapperLayer{:layer}
   layer
end

function (l::EdgeEmbed)(X::ETGraph, ps, st)
   Φ2, st = l.layer(X.edge_data, ps, st)
   Φ3 = reshape_embedding(Φ2, X)
   return Φ3, st
end

function evaluate_ed(l::EdgeEmbed, X::ETGraph, ps, st)
   (Φ2, ∂Φ2), st = evaluate_ed(l.layer, X.edge_data, ps, st)
   Φ3 = reshape_embedding(Φ2, X)
   ∂Φ3 = reshape_embedding(∂Φ2, X)
   return (Φ3, ∂Φ3), st
end

# NOTE: this should not need an rrule because l.layer should have an rrule
#       and reshape_embedding already has an rrule defined.
#       for some manual test implementations the following is still useful.

function _pullback_edge_embedding(∂Φ3, dΦ3, X::ETGraph)
   ∂Φ2 = rev_reshape_embedding(∂Φ3, X)
   dΦ2 = rev_reshape_embedding(dΦ3, X)
   return dropdims( sum(∂Φ2 .* dΦ2, dims = 2), dims = 2)
end
