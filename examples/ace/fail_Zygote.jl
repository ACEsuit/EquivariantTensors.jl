import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random, Test
using Zygote, LuxCore, Lux
import Optimisers as OPT
import ForwardDiff as FDiff 

@info("Preliminary Pullback test for lux ace model")

##
struct DotL <: Lux.AbstractLuxLayer 
   nin::Int 
end

LuxCore.initialparameters(rng::AbstractRNG, l::DotL) = 
      (w = randn(rng, l.nin) * 0.1, )

LuxCore.initialstates(rng::AbstractRNG, l::DotL) = 
      NamedTuple()

(l::DotL)(x, ps, st) = sum(ps.w .* x), st

## 
Dtot = 8
maxl = 6
ORD = 3 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = (0, 1), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

model = Chain(; 
      embed = Parallel(nothing; 
               Rnl = Chain( WrappedFunction(𝐫 -> norm.(𝐫)),  
                            P4ML.lux(rbasis) ), 
               Ylm = P4ML.lux(ybasis)),
      𝔹 = 𝔹basis, 
      y01 = Parallel(nothing; 
            y0 = DotL(length(𝔹basis, 0)), 
            y1 = DotL(length(𝔹basis, 1)), ), 
      iml = WrappedFunction(x -> (exp(im * x[1]) * x[1], exp.(im * x[2]) .* x[2])),
      out = WrappedFunction(x -> real(x[1] + sum(abs2, x[2]) ))
      )

##

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()
nX = 7
𝐫 = [ rand_x() for _ = 1:nX ]

rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)
φ, _ = Lux.apply(model, 𝐫, ps, st)

# ==========================
# Pullback Test (should succeed)
# ==========================
val, pb = Zygote.pullback(𝐫 -> Lux.apply(model, 𝐫, ps, st)[1], 𝐫)
pb(val)

# ==========================
# Model Decomposition (Split into parts)
# ==========================
model1 = Chain(;
    embed = Parallel(nothing;
        Rnl = Chain(
            WrappedFunction(𝐫 -> norm.(𝐫)),
            P4ML.lux(rbasis)
        ),
        Ylm = P4ML.lux(ybasis)
    ),
    𝔹 = 𝔹basis
)

ps1, st1 = Lux.setup(rng, model1)
φ1, _ = Lux.apply(model1, 𝐫, ps1, st1)

model2 = Chain(;
    y01 = Parallel(nothing;
        y0 = DotL(length(𝔹basis, 0)),
        y1 = DotL(length(𝔹basis, 1))
    ),
    iml = WrappedFunction(x -> (
        exp(im * x[1]) * x[1],
        exp.(im * x[2]) .* x[2]
    )),
    out = WrappedFunction(x -> real(x[1] + sum(abs2, x[2])))
)

ps2, st2 = Lux.setup(rng, model2)
φ2, _ = Lux.apply(model2, φ1, ps2, st2)


# ==========================
# Backward Pass (model2)
# ==========================
val2, pb2 = Zygote.pullback(φ1 -> Lux.apply(model2, φ1, ps2, st2)[1], φ1)
∂BB = pb2(val2)[1]
@show typeof(∂BB)

# ==========================
# Backward Pass (model1)
# ==========================
val1, pb1 = Zygote.pullback(𝐫 -> Lux.apply(model1, 𝐫, ps1, st1)[1], 𝐫)
pb1(val1)         # should succeed
pb1(∂BB)          # should succeed
