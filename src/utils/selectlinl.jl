

"""
   struct SelectLinL <: AbstractLuxLayer

A Lux layer which acts as a simple linear layer, but using a categorical 
variable to select a weight matrix:
```
   P -> W[x] * P 
```
This layer is experimental and likely very inefficient. 
"""
struct SelectLinL{TSEL} <: AbstractLuxLayer
   in_dim::Int
   out_dim::Int
   ncat::Int
   selector::TSEL
end

LuxCore.initialstates(rng::AbstractRNG, l::SelectLinL) = NamedTuple()

function LuxCore.initialparameters(rng::AbstractRNG, l::SelectLinL) 
   W = randn(rng, l.out_dim, l.in_dim, l.ncat) * sqrt(2 / (l.in_dim + l.out_dim))
   return (W = W,)
end

(l::SelectLinL)( P_X, ps, st) = 
      _apply_selectlinl(l, P_X[1], P_X[2], ps, st)

# TODO: this case distinction is a hack; it would be better to 
#       use types one cannot iterate over. Something to look into. 
#       and an argument to move to DecoratedParticles.jl

function _apply_selectlinl(l, P, X::AbstractArray, ps, st)
   B = reduce(vcat, transpose(ps.W[:, :, l.selector(x)] * P[i, :])
                    for (i, x) in enumerate(X) )
   return B, st 
end

function _apply_selectlinl(l, P, x::Union{NamedTuple, Number, StaticArray},
                           ps, st)
   B = ps.W[:, :, l.selector(x)] * P
   return B, st
end


# ---------------------------------
#  ChainRulesCore rrule for SelectLinL callable (Matrix P, AbstractArray X version)
import ChainRulesCore: rrule, NoTangent, @thunk

# rrule for SelectLinL when called as a Lux layer: l(P_X, ps, st)
# where P_X = (P, X) is a tuple/namedtuple with P = matrix, X = array of selectors
function rrule(l::SelectLinL, P_X, ps, st)
   P_raw = P_X[1]
   X = P_X[2]

   # Handle case where P might be wrapped in a tuple (from SparseACEbasis)
   P = if P_raw isa Tuple
      P_raw[1]  # Unwrap the tuple
   else
      P_raw
   end
   is_P_wrapped = P_raw isa Tuple

   # Forward pass - P should be a matrix (n_items x in_dim), X is array of selectors
   nX = length(X)
   out_dim = l.out_dim
   in_dim = l.in_dim

   # Sanity check
   if !(P isa AbstractMatrix)
      error("SelectLinL rrule expected P to be a Matrix, got $(typeof(P))")
   end

   # Compute forward pass and save intermediate selections
   selectors = [l.selector(x) for x in X]

   # P is (nX x in_dim), each row is one feature vector
   B_rows = [transpose(ps.W[:, :, selectors[i]] * P[i, :]) for i in 1:nX]
   B = reduce(vcat, B_rows)

   function selectlinl_pb(∂out)
      ∂B = ∂out[1]  # Gradient of output B
      # ∂out[2] is for st, which should be NoTangent()

      # Initialize gradients
      ∂P = zeros(eltype(P), size(P))
      ∂W = zeros(eltype(ps.W), size(ps.W))

      # ∂B is (nX, out_dim) -> each row corresponds to one x
      for i in 1:nX
         sel = selectors[i]
         W_sel = ps.W[:, :, sel]  # (out_dim, in_dim)
         ∂B_i = ∂B[i, :]  # (out_dim,)

         # Forward: B_i = W_sel * P[i, :]
         # ∂P[i, :] = W_sel' * ∂B_i
         # ∂W[:, :, sel] += ∂B_i * P[i, :]'
         ∂P[i, :] = W_sel' * ∂B_i
         ∂W[:, :, sel] += ∂B_i * P[i, :]'
      end

      ∂ps = (W = ∂W,)
      # Return gradients for: l, P_X, ps, st
      # For P_X we need to return a tangent for the tuple (P, X)
      # X contains categorical data so its tangent is NoTangent()
      # If P was wrapped in a tuple, we need to wrap the gradient too
      ∂P_tangent = is_P_wrapped ? (∂P,) : ∂P
      ∂P_X = ChainRulesCore.Tangent{typeof(P_X)}(∂P_tangent, NoTangent())
      return NoTangent(), ∂P_X, ∂ps, NoTangent()
   end

   return (B, st), selectlinl_pb
end
