
using ConcreteStructs
using LuxCore: AbstractLuxContainerLayer
using LinearAlgebra: dot


# -------------------------------------------------------------------
#
# StateEmbed: embed a particle state into a vector — apply a transform to the
# state (→ a number / SVectors), then a basis (or other layer), with a custom
# `evaluate_ed` that differentiates through the (XState) input (needed for
# jacobians). `EmbedDP` is a deprecated alias.
#
# (`EdgeEmbed`, the graph adapter, now lives in `src/graphs/edgeembed.jl`.)


"""
   struct StateEmbed

Embed a particle state into a vector: apply a transform to the state (→ a number
/ SVectors), then evaluate a basis (or other layer) on the transformed state.
Essentially a 3-stage `Chain` (`trans → basis → post`), but with a custom
`evaluate_ed` that differentiates through an XState input — needed e.g. for
jacobians. `EmbedDP` is a deprecated alias.
"""
@concrete struct StateEmbed <: AbstractLuxContainerLayer{(:trans, :basis, :post)}
   trans
   basis
   post
end

StateEmbed(trans, basis) = StateEmbed(trans, basis, IDpost())


(l::StateEmbed)(X::AbstractArray, ps, st) = _apply_stateembed(l, X, ps, st), st

function _apply_stateembed(l::StateEmbed, X::AbstractArray, ps, st)
   # first gets rid of the state variable in the return
   # NOTE: here we assume that the transform implicitly broadcasts
   Y, _ = l.trans(X, ps.trans, st.trans)
   P2, _ = l.basis(Y, ps.basis, st.basis)
   # opportunity for another transformation that depends also on X
   # if post == IDpost then this is a no-op
   post_P2, _ = l.post((P2, X), ps.post, st.post)
   return post_P2
end


function evaluate_ed(l::StateEmbed, X::AbstractArray, ps, st)
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


# StateEmbed is using low-dimensional input, high-dimensional output
# hence forward-mode differentiation is preferred. This is provided
# by the following rrule. But it should be tested at some point
# whether this is really efficient. No rush for the moment, the scaling
# at least ought to be ok.
#
# TODO: write tests for this
#
function rrule(::typeof(_apply_stateembed),
               l::StateEmbed, X::AbstractArray, ps, st)

   (P, dP), st = evaluate_ed(l, X, ps, st)

   function _pb_stateembed(_∂P)
      ∂P = unthunk(_∂P)
      ∂X = dropdims( sum(∂P .* dP, dims = 2), dims = 2)
      return NoTangent(), NoTangent(), ∂X, NoTangent(), NoTangent()
   end

   return P, _pb_stateembed
end


# -------------------------------------------------------------------

struct IDpost <: AbstractLuxLayer
end

(l::IDpost)(P_X, ps, st) = P_X[1], st

pfwd_ed(l::IDpost, P_dP_X, ps, st) = (P_dP_X[1], P_dP_X[2]), st


# --- deprecated alias (pre-2026-06 name) ---
Base.@deprecate_binding EmbedDP StateEmbed false
