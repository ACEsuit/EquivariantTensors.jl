
# Convert a `LearnableRnlBasis` into a parameter-free `SplineRnlBasis`.
# Ported from ACEpotentials `src/models/Rnl_splines.jl::splinify`, with the
# `Interpolations.cubic_spline_interpolation` call replaced by
# `Polynomials4ML.splinify` + `Polynomials4ML.CubicSplines` (which uses
# Interpolations internally to obtain C2,2 regular splines, so ET does not
# need a direct dependency on Interpolations / OffsetArrays).

splinify(basis::SplineRnlBasis, ps; kwargs...) = basis


function splinify(basis::LearnableRnlBasis, ps; nnodes = 100)

   # transform : r ∈ [rin, rcut] -> x
   # and then Rnl =  Wnl_q * Pq(x) * env(x) gives the basis.
   # The problem with this is that we cannot evaluate the envelope from just
   # r coordinates. We therefore keep the transform inside the splinified
   # basis and only splinify the last operation, x -> Rnl(x).
   # this also has the potential advantage that few spline points are needed,
   # and that we get access to the same meta-information about the model building.
   #
   # in the following we assume all transforms map [rin, rcut] -> [-1, 1]

   NZ = _get_nz(basis)
   polys = basis.polys

   _splines = [ _splinify_ij(polys, (@view ps.Wnlq[:, :, iz0, iz1]), nnodes)
                for iz0 = 1:NZ, iz1 = 1:NZ ]
   splines = SMatrix{NZ, NZ}(_splines)

   spl_basis = SplineRnlBasis(basis._i2z,
                              basis.transforms,
                              basis.envelopes,
                              splines,
                              basis.rin0cuts,
                              basis.spec,
                              basis.meta)

   spl_basis.meta["info"] = "constructed from LearnableRnlBasis via `splinify`"

   # we should probably store more meta-data from which the splines can be
   # easily reconstructed.

   return spl_basis
end


# spline the map x -> Wnlq_ij * Pq(x) on [-1, 1] using P4ML CubicSplines.
function _splinify_ij(polys, Wnlq_ij, nnodes)
   LEN = size(Wnlq_ij, 1)
   f = x -> SVector{LEN}(Wnlq_ij * P4ML.evaluate(polys, x))
   return P4ML.splinify(f, -1.0, 1.0, nnodes)
end
