#
# TODO: rethink EdgeEmbed and variants
#

using ConcreteStructs 
using Lux: AbstractLuxWrapperLayer, AbstractLuxContainerLayer
using LinearAlgebra: dot 
import Lux 


# -------------------------------------------------------------------
#
# alternative, maybe simpler embedding layer that utilizes more 
# of the Lux infrastructure / automatic differentiation.
#

"""
   struct EdgeEmbed 

Wraps a layer that embeds an edge state into Vector to manage the 
reformatting of the embedding from a list of embedded states into 
a 3-dimensionsonal tensor that is aware of the graph structure.
"""
@concrete struct EdgeEmbed <: AbstractLuxWrapperLayer{:layer}
   layer
   name
end

EdgeEmbed(layer; name = "Edge Embedding") = EdgeEmbed(layer, name)

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


# -------------------------------------------------------------------


"""
   struct EmbedDP 

Embed a particle state into a vector. This is done by first applying a 
transform to the particle state into a number of SVector and then evaluating 
the basis (or other layer) on the transformed state. 

This is basically a 2-stage Chain, but with additional logic, specifically 
the implementation of evaluate_ed allowing differentiation through 
and XState or NamedTuple input. 
"""
@concrete struct EmbedDP <: AbstractLuxContainerLayer{(:trans, :basis)}
   trans
   basis
   name
end

EmbedDP(trans, basis; name = "") = 
         EmbedDP(trans, basis, name)

Base.show(io::IO, ::MIME"text/plain", l::EmbedDP) = 
      print(io, "EmbedDP($(l.name))")         


function (l::EmbedDP)(X::AbstractArray, ps, st)
   # first gets rid of the state variable in the return 
   Y = broadcast(first ∘ l.trans, X, Ref(ps.trans), Ref(st.trans))
   Φ2, _ = l.basis(Y, ps.basis, st.basis)
   return Φ2, st
end

function evaluate_ed(l::EmbedDP, X::AbstractArray, ps, st)
   # first gets rid of the state variable in the return 
   ftrans = _x -> first(l.trans(_x, ps.trans, st.trans))
   Y = broadcast(ftrans, X,)
   Φ2, dΦ2 = evaluate_ed(l.basis, Y, ps.basis, st.basis)

   # pullback through the transform to get ∂Φ2
   _pb1(x, dφ) = DiffNT.grad_fd(_x -> dot(ftrans(_x), dφ), x)
   ∂Φ2 = broadcast(_pb1, X, dΦ2)
   
   return (Φ2, ∂Φ2), st
end


# TODO: 
#   definitely still need to implement the rrule for this layer 
#   it can utilize evaluate_ed for efficient pullback. 

# function rrule(::typeof(reshape_embedding), ϕ2, X::ETGraph)
#    ϕ3 = reshape_embedding(ϕ2, X)

#    function _pb_ϕ(∂ϕ3)
#       ∂ϕ2 = rev_reshape_embedding(∂ϕ3, X)
#       return NoTangent(), ∂ϕ2, NoTangent()
#    end

#    return ϕ3, _pb_ϕ
# end

