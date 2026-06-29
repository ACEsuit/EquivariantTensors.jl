# Regression test for issue #135: `pullback` for a SparseACEbasis must be
# type-stable (it previously returned `Tuple{Any, Any}` whenever the cotangent
# `∂BB` was a `Vector`, because `whatalloc` splatted a runtime `Vector{DataType}`
# into `promote_type`).

import Polynomials4ML as P4ML
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random, Test

@info("Testing pullback type stability (issue #135)")

function _build_tensor(LL)
   Dtot, maxl, ORD = 8, 6, 3
   rbasis = P4ML.legendre_basis(Dtot + 1); Rn_spec = P4ML.natural_indices(rbasis)
   ybasis = P4ML.real_sphericalharmonics(maxl); Ylm_spec = P4ML.natural_indices(ybasis)
   nnll = ET.sparse_nnll_set(; ORD = ORD, minn = 0, maxn = Dtot, maxl = maxl,
               level = bb -> sum((b.n + b.l) for b in bb; init = 0), maxlevel = Dtot)
   tensor = ET.sparse_equivariant_tensors(; LL = LL, mb_spec = nnll,
               Rnl_spec = Rn_spec, Ylm_spec = Ylm_spec, basis = real)
   return tensor, length(Rn_spec), length(Ylm_spec)
end

function test_pullback_typestable(LL; nX = 7, rng = Random.MersenneTwister(1234))
   tensor, nR, nY = _build_tensor(LL)
   Rnl = randn(rng, nX, nR)
   Ylm = randn(rng, nX, nY)
   A  = ET.evaluate(tensor.abasis, (Rnl, Ylm))
   BB = ET.evaluate(tensor, Rnl, Ylm, NamedTuple(), NamedTuple())   # Tuple of blocks

   # (1) Tuple cotangent — the Zygote rrule path (possibly heterogeneous L blocks)
   rt = only(Base.return_types(ET.pullback, typeof.((BB, tensor, Rnl, Ylm, A))))
   @test isconcretetype(rt)
   @test (@inferred ET.pullback(BB, tensor, Rnl, Ylm, A)) isa Tuple{<:Matrix, <:Matrix}

   # (2) Vector cotangent of a single block — the force-evaluation path (#135)
   ∂B = [BB[1]]
   rtv = only(Base.return_types(ET.pullback, typeof.((∂B, tensor, Rnl, Ylm, A))))
   @test isconcretetype(rtv)
   @test (@inferred ET.pullback(∂B, tensor, Rnl, Ylm, A)) isa Tuple{<:Matrix, <:Matrix}
end

for LL in ((0,), (0, 1))
   test_pullback_typestable(LL)
end
