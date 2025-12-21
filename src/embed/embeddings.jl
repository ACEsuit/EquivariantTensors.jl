
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

Also implements an evaluate_ed wrapper, which is useful for 
computing jacobians. 
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
@concrete struct EmbedDP <: AbstractLuxContainerLayer{(:trans, :basis, :post)}
   trans
   basis
   post 
end

EmbedDP(trans, basis) = EmbedDP(trans, basis, IDpost())

# Base.show(io::IO, ::MIME"text/plain", l::EmbedDP) = 
#       print(io, "EmbedDP($(l.name))")         


(l::EmbedDP)(X::AbstractArray, ps, st) = _apply_embeddp(l, X, ps, st), st 

function _apply_embeddp(l::EmbedDP, X::AbstractArray, ps, st)   
   # first gets rid of the state variable in the return 
   # NOTE: here we assume that the transform implicitly broadcasts
   Y, _ = l.trans(X, ps.trans, st.trans)
   P2, _ = l.basis(Y, ps.basis, st.basis)
   # opportunity for another transformation that depends also on X 
   # if post == IDpost then this is a no-op
   post_P2, _ = l.post((P2, X), ps.post, st.post)
   return post_P2
end

#
# TODO: to make this GPU compatible the pullback 
#       needs to be broadcast implicitly, otherwise 
#       we have some hellishly weird type issues 
#       in the GPU kernels... 
#
import DecoratedParticles: vstate_type 

function evaluate_ed(l::EmbedDP, X::AbstractArray, ps, st)
   Y, _ = l.trans(X, ps.trans, st.trans)
   P2, dP2 = evaluate_ed(l.basis, Y, ps.basis, st.basis)

   # pushforward the P' through the post-transform layer 
   # if post == IDpost then this is a no-op
   (pP2, d_pP2), _ = pfwd_ed(l.post, (P2, dP2, X), ps.post, st.post)

   # pullback through the transform to get ∂P2
   # this is kind of a temporary hack and we need to understand why 
   # there is no simple generic solution ... 
   ∂_pP2 = _pb_ed(l.trans, d_pP2, X, ps.trans, st.trans)
   
   return (pP2, ∂_pP2), st
end


# EmbedDP is using low-dimensional input, high-dimensional output 
# hence forward-mode differentiation is preferred. This is provided 
# by the following rrule. 
#
# TODO: write tests for this 
#
function rrule(::typeof(_apply_embeddp), 
               l::EmbedDP, X::AbstractArray, ps, st)

   (P, dP), st = evaluate_ed(l, X, ps, st)

   function _pb_embeddp(_∂P)
      ∂P = unthunk(_∂P)
      ∂X = dropdims( sum(∂P .* dP, dims = 2), dims = 2) 
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end

   return P, _pb_embeddp
end




# -------------------------------------------------------------------

struct IDpost <: AbstractLuxLayer
end 

(l::IDpost)(P_X, ps, st) = P_X[1], st 

pfwd_ed(l::IDpost, P_dP_X, ps, st) = (P_dP_X[1], P_dP_X[2]), st


