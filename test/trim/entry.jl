# juliac --trim=safe entry point exercising the trim-safe construction API.
# test/trim/build_and_run.jl compiles this file and compares the executable's
# output with checksum_lines() evaluated by ordinary Julia.
import EquivariantTensors as ET
using SparseArrays: findnz
include(joinpath(@__DIR__, "..", "test_utils", "trim_cases.jl"))

_mix(h::UInt64, x::UInt64) = (h ⊻ x) * 0x100000001b3

function checksum_lines()::Vector{String}
   out = String[]
   for (maxn, maxl, ord) in TRIM_CASES
      mb = trim_mb_spec(maxn, maxl, ord); R = trim_rnl(maxn, maxl); Y = trim_ylm(maxl)
      t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      I, J, V = findnz(t.symm)
      h = 0xcbf29ce484222325
      for k in eachindex(V)
         h = _mix(h, UInt64(I[k])); h = _mix(h, UInt64(J[k])); h = _mix(h, reinterpret(UInt64, V[k]))
      end
      for (r, y) in t.Aspec_raw
         h = _mix(h, UInt64(r)); h = _mix(h, UInt64(y))
      end
      for bb in t.𝔸spec, b in bb
         h = _mix(h, UInt64(b.n)); h = _mix(h, UInt64(b.l)); h = _mix(h, UInt64(b.m + 1024))
      end
      push!(out, string(size(t.symm, 1), " ", size(t.symm, 2), " ", length(V), " ", h))
   end
   return out
end

function (@main)(args::Vector{String})::Cint
   for line in checksum_lines()
      println(Core.stdout, line)
   end
   return 0
end
