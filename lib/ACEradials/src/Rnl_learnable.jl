
# Learnable radial basis. Ported from ACEpotentials `src/models/Rnl_basis.jl`
# and `Rnl_learnable.jl`, with the type renamed `LearnableRnlrzzBasis` ->
# `LearnableRnlBasis` and the `Interpolations`-based spline machinery removed
# (see `Rnl_splines.jl` / `splinify.jl`).
#
# NOTE: each SMatrix in the Rnl types indexes (i, j) where i is the center
# atom and j the neighbour.

const NT_RIN0CUTS{T} = NamedTuple{(:rin, :r0, :rcut), Tuple{T, T, T}}
const NT_NL_SPEC = NamedTuple{(:n, :l), Tuple{Int, Int}}


struct LearnableRnlBasis{NZ, TPOLY, TT, TENV, T} <: AbstractLuxLayer
   _i2z::NTuple{NZ, Int}
   polys::TPOLY
   transforms::SMatrix{NZ, NZ, TT}
   envelopes::SMatrix{NZ, NZ, TENV}
   # --------------
   rin0cuts::SMatrix{NZ, NZ, NT_RIN0CUTS{T}}  # matrix of (rin, rout, rcut)
   spec::Vector{NT_NL_SPEC}
   # --------------
   # meta
   meta::Dict{String, Any}
end


# a few getter functions for convenient access to those fields of matrices
_rincut_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)]
_rin0cuts_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)]
_rcut_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].rcut
_rin_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].rin
_r0_zz(obj, zi, zj) = obj.rin0cuts[_z2i(obj, zi), _z2i(obj, zj)].r0
_envelope_zz(obj, zi, zj) = obj.envelopes[_z2i(obj, zi), _z2i(obj, zj)]
_transform_zz(obj, zi, zj) = obj.transforms[_z2i(obj, zi), _z2i(obj, zj)]

_get_T(basis::LearnableRnlBasis) = typeof(basis.rin0cuts[1,1].rin)


# ------------------------------------------------------------
#      CONSTRUCTORS AND UTILITIES
# ------------------------------------------------------------

function LearnableRnlBasis(
            zlist, polys, transforms, envelopes, rin0cuts,
            spec::AbstractVector{NT_NL_SPEC};
            meta=Dict{String, Any}())
   NZ = length(zlist)
   LearnableRnlBasis(_convert_zlist(zlist),
                     polys,
                     _make_smatrix(transforms, NZ),
                     _make_smatrix(envelopes, NZ),
                     # --------------
                     _make_smatrix(rin0cuts, NZ),
                     collect(spec),
                     meta)
end

Base.length(basis::LearnableRnlBasis) = length(basis.spec)

function initialparameters(rng::AbstractRNG,
                           basis::LearnableRnlBasis)
   NZ = _get_nz(basis)
   len_nl = length(basis)
   len_q = length(basis.polys)

   T = _get_T(basis)
   Wnlq = zeros(T, len_nl, len_q, NZ, NZ)
   for i = 1:NZ, j = 1:NZ
      Wnlq[:, :, i, j] .= glorot_normal(rng, T, len_nl, len_q)
   end

   return (Wnlq = Wnlq, )
end

function initialstates(rng::AbstractRNG,
                       basis::LearnableRnlBasis)
   return NamedTuple()
end


function parameterlength(basis::LearnableRnlBasis)
   NZ = _get_nz(basis)
   len_nl = length(basis)
   len_q = length(basis.polys)
   return len_nl * len_q * NZ * NZ
end


# ------------------------------------------------------------
#      EVALUATION INTERFACE
# ------------------------------------------------------------

(l::LearnableRnlBasis)(args...) = evaluate(l, args...)

function evaluate!(Rnl, basis::LearnableRnlBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = @view ps.Wnlq[:, :, iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = P4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, r, x)
   Rnl[:] .= Wij * (P .* e)
   return Rnl
end

function evaluate(basis::LearnableRnlBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   Wij = @view ps.Wnlq[:, :, iz, jz]
   trans_ij = basis.transforms[iz, jz]
   x = trans_ij(r)
   P = P4ML.evaluate(basis.polys, x)
   env_ij = basis.envelopes[iz, jz]
   e = evaluate(env_ij, r, x)
   return Wij * (P .* e)
end


function evaluate_batched!(Rnl,
                           basis::LearnableRnlBasis,
                           rs, zi, zjs, ps, st)

   @assert length(rs) == length(zjs)
   @assert size(Rnl, 1) >= length(rs)
   @assert size(Rnl, 2) >= length(basis)

   for j = 1:length(rs)
      iz = _z2i(basis, zi)
      jz = _z2i(basis, zjs[j])
      trans_ij = basis.transforms[iz, jz]
      x = trans_ij(rs[j])
      env_ij = basis.envelopes[iz, jz]
      e = evaluate(env_ij, rs[j], x)
      P = P4ML.evaluate(basis.polys, x) .* e
      Rnl[j, :] = (@view ps.Wnlq[:, :, iz, jz]) * P
   end

   return Rnl
end

function whatalloc(::typeof(evaluate_batched!),
                    basis::LearnableRnlBasis,
                    rs::AbstractVector{T}, zi, zjs, ps, st) where {T}
   T1 = promote_type(eltype(ps.Wnlq), T)
   return (T1, length(rs), length(basis))
end

function evaluate_batched(basis::LearnableRnlBasis,
                          rs, zi, zjs, ps, st)
   Rnl = zeros(whatalloc(evaluate_batched!, basis, rs, zi, zjs, ps, st)...)
   return evaluate_batched!(Rnl, basis, rs, zi, zjs, ps, st)
end

# ----- gradients
# because the typical scenario is that we have few r, then moderately
# many q and then many (n, l), this seems to be best done in Forward-mode.
# in initial tests it seems the performance is very near optimal
# so there is little sense trying to do something manual.

function evaluate_ed(basis::LearnableRnlBasis, r::T, Zi, Zj, ps, st) where {T <: Real}
   d_r = Dual{T}(r, one(T))
   d_Rnl = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl)
   return Rnl, Rnl_d
end


function evaluate_ed_batched!(Rnl, Rnl_d,
                             basis::LearnableRnlBasis,
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}

   @assert length(rs) == length(Zs)
   for j = 1:length(rs)
      d_r = Dual{T}(rs[j], one(T))
      d_Rnl = evaluate(basis, d_r, Zi, Zs[j], ps, st)  # should reuse memory here
      for t = 1:size(Rnl, 2)
         Rnl[j, t] = ForwardDiff.value(d_Rnl[t])
         Rnl_d[j, t] = ForwardDiff.extract_derivative(T, d_Rnl[t])
      end
   end

   return Rnl, Rnl_d
end

function whatalloc(::typeof(evaluate_ed_batched!),
                    basis::LearnableRnlBasis,
                    rs::AbstractVector{T}, Zi, Zs, ps, st) where {T}
   T1 = promote_type(eltype(ps.Wnlq), T)
   return (T1, length(rs), length(basis)), (T1, length(rs), length(basis))
end

function evaluate_ed_batched(basis::LearnableRnlBasis,
                        rs::AbstractVector{T}, Zi, Zs, ps, st
                        ) where {T <: Real}
   allocinfo = whatalloc(evaluate_ed_batched!, basis, rs, Zi, Zs, ps, st)
   Rnl = zeros(allocinfo[1]...)
   Rnl_d = zeros(allocinfo[2]...)
   return evaluate_ed_batched!(Rnl, Rnl_d, basis, rs, Zi, Zs, ps, st)
end



# -------- RRULES

# NB : iz = īz = _z2i(z0) throughout
#
# Rnl[j, nl] = Wnlq[iz, jz] * Pq * e
# ∂_Wn̄l̄q̄[īz,j̄z] { ∑_jnl Δ[j,nl] * Rnl[j, nl] }
#    = ∑_jnl Δ[j,nl] * Pq * e * δ_q̄q * δ_l̄l * δ_n̄n * δ_{īz,iz} * δ_{j̄z,jz}
#    = ∑_{jz = j̄z} Δ[j̄z, n̄l̄] * P_q̄ * e
#
function pullback_evaluate_batched(Δ, basis::LearnableRnlBasis,
                                   rs, zi, zjs, ps, st)
   @assert length(rs) == length(zjs)
   # Δ may arrive as a (Inplaceable)Thunk, e.g. from the sum(abs2, _) rule
   Δ = unthunk(Δ)

   # output storage for the gradients
   T_∂Wnlq = promote_type(eltype(Δ), eltype(rs))
   NZ = _get_nz(basis)
   ∂Wnlq = zeros(T_∂Wnlq, size(ps.Wnlq))

   # then evaluate the rest in-place
   for j = 1:length(rs)
      iz = _z2i(basis, zi)
      jz = _z2i(basis, zjs[j])
      trans_ij = basis.transforms[iz, jz]
      x = trans_ij(rs[j])
      env_ij = basis.envelopes[iz, jz]
      e = evaluate(env_ij, rs[j], x)
      P = P4ML.evaluate(basis.polys, x) .* e
      # TODO: the P shouuld be stored inside a closure in the
      #       forward pass and then resused.

      # TODO:  ... and obviously this part here needs to be moved
      # to a SIMD loop.
      ∂Wnlq[:, :, iz, jz] .+= Δ[j, :] * P'
   end

   return (Wnlq = ∂Wnlq,)
end


# NOTE: only the gradient w.r.t. the parameters `ps` is implemented here;
# the tangent w.r.t. `rs` is not (derivatives w.r.t. positions go through
# the `evaluate_ed_batched` path).
function rrule(::typeof(evaluate_batched),
               basis::LearnableRnlBasis,
               rs, zi, zjs, ps, st)
   Rnl = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return Rnl, Δ -> (NoTangent(), NoTangent(), NoTangent(),
                     NoTangent(), NoTangent(),
                     pullback_evaluate_batched(Δ, basis, rs, zi, zjs, ps, st),
                     NoTangent())
end
