
# Splined radial basis. Ported from ACEpotentials `src/models/Rnl_basis.jl`
# and `Rnl_splines.jl`, with the type renamed `SplineRnlrzzBasis` ->
# `SplineRnlBasis` and the internal spline storage switched from
# `Interpolations` to `Polynomials4ML.CubicSplines` (see `splinify.jl`).

struct SplineRnlBasis{NZ, TT, TENV, NX, LEN, T} <: AbstractLuxLayer
   _i2z::NTuple{NZ, Int}
   transforms::SMatrix{NZ, NZ, TT}
   envelopes::SMatrix{NZ, NZ, TENV}
   splines::SMatrix{NZ, NZ, P4ML.CubicSplines{NX, LEN, T}}
   # --------------
   rin0cuts::SMatrix{NZ, NZ, NT_RIN0CUTS{T}}  # matrix of (rin, rout, rcut)
   spec::Vector{NT_NL_SPEC}
   # --------------
   meta::Dict{String, Any}
end


# evaluate a single P4ML CubicSplines basis at a scalar `x`; the number of
# nodes `NX` is carried in the type so this stays type stable.
_eval_spline(spl::P4ML.CubicSplines{NX}, x) where {NX} =
      P4ML._eval_cubspl(x, spl.F, spl.G, spl.x0, spl.x1, NX)


# ------------------------------------------------------------
#      CONSTRUCTORS AND UTILITIES
# ------------------------------------------------------------

Base.length(basis::SplineRnlBasis) = length(basis.spec)

function initialparameters(rng::AbstractRNG,
                           basis::SplineRnlBasis)
   return NamedTuple()
end

function initialstates(rng::AbstractRNG,
                       basis::SplineRnlBasis)
   return NamedTuple()
end


# ------------------------------------------------------------
#      EVALUATION INTERFACE
# ------------------------------------------------------------

(l::SplineRnlBasis)(args...) = evaluate(l, args...)


function evaluate(basis::SplineRnlBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   T_ij = basis.transforms[iz, jz]
   env_ij = basis.envelopes[iz, jz]
   spl_ij = basis.splines[iz, jz]

   x_ij = T_ij(r)
   e_ij = evaluate(env_ij, r, x_ij)

   return _eval_spline(spl_ij, x_ij) * e_ij
end


function evaluate_batched!(Rnl, basis::SplineRnlBasis,
                           rs, zi, zjs, ps, st)

   @assert length(rs) == length(zjs)
   # evaluate the first one to get the types and size
   Rnl_1 = evaluate(basis, rs[1], zi, zjs[1], ps, st)
   # ... and then store it
   Rnl[1, :] .= Rnl_1

   # then evaluate the rest in-place
   for j = 2:length(rs)
      Rnl[j, :] = evaluate(basis, rs[j], zi, zjs[j], ps, st)
   end

   return Rnl
end

function whatalloc(::typeof(evaluate_batched!),
                   basis::SplineRnlBasis,
                   rs, zi, zjs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis))
end


function evaluate_batched(basis::SplineRnlBasis,
                           rs, zi, zjs, ps, st)
   Rnl = zeros(whatalloc(evaluate_batched!, basis, rs, zi, zjs, ps, st)...)
   return evaluate_batched!(Rnl, basis, rs, zi, zjs, ps, st)
end

# ----- gradients
# because the typical scenario is that we have few r, then moderately
# many q and then many (n, l), this seems to be best done in Forward-mode.

function evaluate_ed(basis::SplineRnlBasis, r::T, Zi, Zj, ps, st) where {T <: Real}
   d_r = Dual{T}(r, one(T))
   d_Rnl = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl)
   return Rnl, Rnl_d
end



function evaluate_ed_batched!(Rnl, Rnl_d,
                             basis::SplineRnlBasis,
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}

   @assert length(rs) == length(Zs)
   for j = 1:length(rs)
      Rnl_j, ∇Rnl_j = evaluate_ed(basis, rs[j], Zi, Zs[j], ps, st)
      Rnl[j, :] = Rnl_j
      Rnl_d[j, :] = ∇Rnl_j
   end

   return Rnl, Rnl_d
end


function whatalloc(::typeof(evaluate_ed_batched!),
                  basis::SplineRnlBasis,
                  rs::AbstractVector, Zi, Zs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis)), (T, length(rs), length(basis))
end


function evaluate_ed_batched(basis::SplineRnlBasis,
                             rs::AbstractVector, Zi, Zs, ps, st)
   alc_Rnl, alc_Rnl_d = whatalloc(evaluate_ed_batched!, basis, rs, Zi, Zs, ps, st)
   Rnl = zeros(alc_Rnl...)
   Rnl_d = zeros(alc_Rnl_d...)
   return evaluate_ed_batched!(Rnl, Rnl_d, basis, rs, Zi, Zs, ps, st)
end


function rrule(::typeof(evaluate_batched),
               basis::SplineRnlBasis,
               rs, zi, zjs, ps, st)
   Rnl = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return Rnl, Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                     NoTangent(), NoTangent())
end
