
import Polynomials4ML as P4ML
import SpheriCart
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random, Test
using Zygote, LuxCore, Lux
import Optimisers as OPT
using ACEbase.Testing: println_slim

include(joinpath(@__DIR__(), "..", "test_utils", "luxtestmodels.jl"))   # LTM.DotL
include(joinpath(@__DIR__(), "..", "test_utils", "diffutils.jl"))       # DIFF

@info("Parameter-gradient pullback test for a Lux ACE model (real)")

##
Dtot = 8; maxl = 6; ORD = 3
rbasis = P4ML.legendre_basis(Dtot+1)
ybasis = SpheriCart.SphericalHarmonics(maxl)

nnll_long = ET.sparse_nnll_set(; ORD = ORD, minn = 0, maxn = Dtot, maxl = maxl,
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensors(; LL = (0, 1), mb_spec = nnll_long,
            Rnl_spec = P4ML.natural_indices(rbasis),
            Ylm_spec = P4ML.natural_indices(ybasis), basis = real)

model = Chain(;
      embed = Parallel(nothing;
               Rnl = Chain(WrappedFunction(𝐫 -> norm.(𝐫)), rbasis),
               Ylm = ybasis),
      𝔹 = 𝔹basis,
      y01 = Parallel(nothing; y0 = LTM.DotL(length(𝔹basis, 0)),
                              y1 = LTM.DotL(length(𝔹basis, 1)) ),
      out = WrappedFunction(x -> x[1] + sum(abs2, x[2])) )

##
rand_sphere() = (u = randn(SVector{3, Float64}); u / norm(u))
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
𝐫 = [ rand_x() for _ = 1:7 ]

ps, st = LuxCore.setup(MersenneTwister(1234), model)

## parameter gradient: ForwardDiff vs Zygote (via the shared test utils)
g_fd = DIFF.grad_fd_ps(𝐫, model, ps, st)
g_zy = DIFF.grad_zy_ps(𝐫, model, ps, st)
println_slim(@test OPT.destructure(g_fd)[1] ≈ OPT.destructure(g_zy)[1])
