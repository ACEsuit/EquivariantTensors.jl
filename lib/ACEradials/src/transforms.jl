
# Scalar `r -> x ∈ [-1,1]` transforms for the radial bases. Ported from
# ACEpotentials `src/models/radial_transforms.jl`.
#
# These are the single source of the Agnesi math in ACEradials: the
# species-pair / XState transform in `agnesi_dp.jl` is a thin adapter that
# stores one `NormalizedTransform` per species pair and evaluates it at
# `norm(x.𝐫)` (see agents/radials.md §4, agents/radials_restructure.md §3.2).
# The serialization (`write_dict` / `read_dict`) and `Roots`-based inverse
# transform from the ACEpotentials version are intentionally not ported to
# avoid new dependencies.

import ForwardDiff

struct GeneralizedAgnesiTransform{T}
   p::Int
   q::Int
   a::T
   rin::T
   r0::T
end

(t::GeneralizedAgnesiTransform)(r) = evaluate(t, r)

function evaluate(t::GeneralizedAgnesiTransform{T}, r::Number) where {T}
   if r <= t.rin
      return one(promote_type(T, typeof(r)))
   end
   a, r0, q, p, rin = t.a, t.r0, t.q, t.p, t.rin
   s = (r-t.rin)/(t.r0-t.rin)
   return 1 / (1 + a * s^q / (1 + s^(q-p)))
end

evaluate_d(t::GeneralizedAgnesiTransform, r::Number) =
     ForwardDiff.derivative(r -> evaluate(t, r), r)


# ---------------------------------------------------------------------------

"""
Maps the transform `trans` to the standardized interval [-1, 1]
"""
struct NormalizedTransform{T, TT}
   trans::TT
   yin::T
   ycut::T
   rin::T
   rcut::T
end

function NormalizedTransform(trans, rin::Number, rcut::Number)
   yin = trans(rin)
   ycut = trans(rcut)
   return NormalizedTransform(trans, yin, ycut, rin, rcut)
end


(t::NormalizedTransform)(r) = evaluate(t, r)

function evaluate(t::NormalizedTransform, r::Number)
   y = t.trans(r)
   𝟙 = one(typeof(y))
   return min(max(-𝟙, -𝟙 + 2 * (y - t.yin) / (t.ycut - t.yin)), 𝟙)
end

evaluate_d(t::NormalizedTransform, r::Number) =
         ForwardDiff.derivative(r -> evaluate(t, r), r)

# ---------------------------------------------------------------------------

@doc raw"""
`function agnesi_transform:` constructs a generalized agnesi transform.
```
trans = agnesi_transform(r0, p, q)
```
with `q >= p`. This generates an `AnalyticTransform` object that implements
```math
   x(r) = \frac{1}{1 + a (r/r_0)^q / (1 + (r/r0)^(q-p))}
```
with default `a` chosen such that $|x'(r)|$ is maximised at $r = r_0$. But
`a` may also be specified directly as a keyword argument.

The transform satisfies
```math
   x(r) \sim \frac{1}{1 + a (r/r_0)^p} \quad \text{as} \quad r \to 0
   \quad \text{and}
   \quad
   x(r) \sim \frac{1}{1 + a (r/r_0)^p}  \quad \text{as} r \to \infty.
```

As default parameters we recommend `p = 2, q = 4` and the defaults for `a`.
"""
function agnesi_transform(r0, rcut, p, q;
               rin = zero(r0),
               a = (-2 * q + p * (-2 + 4 * q)) / (p + p^2 + q + q^2) )
   @assert p > 0
   @assert q > 0
   @assert q >= p
   @assert a > 0
   @assert 0 < r0 < rcut
   return NormalizedTransform(
                  GeneralizedAgnesiTransform(p, q, a, rin, r0),
                  rin, rcut )
end

function agnesi_transform(rin0cut::NamedTuple, p, q)
   rin = rin0cut.rin
   r0 = rin0cut.r0
   rcut = rin0cut.rcut
   return agnesi_transform(r0, rcut, p, q, rin = rin)
end
