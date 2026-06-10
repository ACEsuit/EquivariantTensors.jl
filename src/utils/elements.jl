
# atomic-number <-> index helpers for multi-species bases.
# These were moved here from ACEpotentials so that the radial bases in
# `src/radials/` can be self-contained. Unlike the ACEpotentials versions
# they do not depend on JuLIP / ACE1x and therefore only accept integer
# atomic numbers.

using StaticArrays: SMatrix

_i2z(obj, i::Integer) = obj._i2z[i]

_get_nz(obj) = length(obj._i2z)

function _z2i(obj, Z)
   for i_Z = 1:length(obj._i2z)
      if obj._i2z[i_Z] == Z
         return i_Z
      end
   end
   error("_z2i : Z = $Z not found in obj._i2z")
   return -1 # never reached
end

# Convert a list of atomic numbers to an `NTuple{NZ, Int}`. Chemical symbols
# must be converted to atomic numbers by the caller (e.g. in ACEpotentials).
function _convert_zlist(zlist)
   all(z -> z isa Integer, zlist) ||
      error("`_convert_zlist` only accepts integer atomic numbers; convert \
             chemical symbols to atomic numbers before calling.")
   return ntuple(i -> convert(Int, zlist[i]), length(zlist))
end

"""
Takes an object and converts it to an `SMatrix{NZ, NZ}` via the following rules:
- if `obj` is already an `SMatrix{NZ, NZ}` then it just return `obj`
- if `obj` is an `AbstractMatrix` and `size(obj) == (NZ, NZ)` then it
   converts it to an `SMatrix{NZ, NZ}` with the same entries.
- otherwise it generates an `SMatrix{NZ, NZ}` filled with the value `obj`.
"""
function _make_smatrix(obj, NZ)
   if obj isa SMatrix{NZ, NZ}
      return obj
   end
   if obj isa AbstractMatrix && size(obj) == (NZ, NZ)
      return SMatrix{NZ, NZ}(obj)
   end
   if obj isa AbstractArray && size(obj) != (NZ, NZ)
      error("`_make_smatrix` : if the input `obj` is an `AbstractArray` \
             then it must be of size `(NZ, NZ)`")
   end
   return SMatrix{NZ, NZ}(fill(obj, (NZ, NZ)))
end
