#
# TODO: rethink EdgeEmbed and variants
#

using ConcreteStructs 
using Lux: AbstractLuxWrapperLayer
import Lux 


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

using Lux: AbstractLuxContainerLayer
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

