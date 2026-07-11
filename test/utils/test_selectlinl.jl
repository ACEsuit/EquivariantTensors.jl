
using EquivariantTensors, StaticArrays, Test, LuxCore, Random
using ACEbase.Testing: println_slim, print_tf
using LinearAlgebra: dot
import Zygote, ForwardDiff
import EquivariantTensors as ET

rng = MersenneTwister(1234)

@info("Testing SelectLinL layer")

##

in_dim, out_dim, ncat = 5, 7, 3
l = ET.SelectLinL(in_dim, out_dim, ncat, identity)
ps, st = LuxCore.setup(rng, l)

##

@info("  construction / initial parameters")

println_slim(@test size(ps.W) == (out_dim, in_dim, ncat))
println_slim(@test eltype(ps.W) == Float64)
println_slim(@test st == NamedTuple())

# initialparameters is rng-deterministic
ps2, _ = LuxCore.setup(MersenneTwister(0), l)
ps3, _ = LuxCore.setup(MersenneTwister(0), l)
println_slim(@test ps2.W == ps3.W)

# Glorot-style scaling: rms(W) ≈ sqrt(2 / (in + out))
rms = sqrt(sum(abs2, ps.W) / length(ps.W))
println_slim(@test isapprox(rms, sqrt(2 / (in_dim + out_dim)); rtol = 0.3))

##

@info("  forward, batched (categories spanning 1:ncat)")

nX = 12
X = [ mod1(i, ncat) for i = 1:nX ]   # identity selector -> category = value
println_slim(@test sort(unique(X)) == collect(1:ncat))

P = randn(nX, in_dim)
B, _ = l((P, X), ps, st)
Bman = reduce(vcat, (ps.W[:, :, X[i]] * P[i, :])' for i = 1:nX)
println_slim(@test size(B) == (nX, out_dim))
println_slim(@test B ≈ Bman)

##

@info("  forward, single input (Number / StaticArray / NamedTuple)")

p = randn(in_dim)

ln = ET.SelectLinL(in_dim, out_dim, ncat, x -> mod1(round(Int, x), ncat))
xn = 2.0
println_slim(@test ET._apply_selectlinl(ln, p, xn, ps.W) ≈
                   ps.W[:, :, ln.selector(xn)] * p)

ls = ET.SelectLinL(in_dim, out_dim, ncat, x -> mod1(round(Int, x[1]), ncat))
xs = SA[2.0, 0.0, 0.0]
println_slim(@test ET._apply_selectlinl(ls, p, xs, ps.W) ≈
                   ps.W[:, :, ls.selector(xs)] * p)

lnt = ET.SelectLinL(in_dim, out_dim, ncat, x -> x.z)
xnt = (z = 3, r = SA[1.0, 2.0, 3.0])
println_slim(@test ET._apply_selectlinl(lnt, p, xnt, ps.W) ≈
                   ps.W[:, :, lnt.selector(xnt)] * p)

##

@info("  reverse-mode: rrule vs ForwardDiff (∂P and ∂W kernels)")

Δ = randn(nX, out_dim)
fP = _P -> dot(Δ, ET._apply_selectlinl(l, _P, X, ps.W))
fW = _W -> dot(Δ, ET._apply_selectlinl(l, P, X, _W))

gP_zy = Zygote.gradient(fP, P)[1]
gP_fd = ForwardDiff.gradient(fP, P)
println_slim(@test gP_zy ≈ gP_fd)

gW_zy = Zygote.gradient(fW, ps.W)[1]
gW_fd = ForwardDiff.gradient(fW, ps.W)
println_slim(@test gW_zy ≈ gW_fd)

# the P::Tuple{AbstractMatrix} rrule variant returns ∂P wrapped in a 1-tuple
res, pb = ET.rrule(ET._apply_selectlinl, l, (P,), X, ps.W)
∂out = pb(Δ)
println_slim(@test res ≈ B)
println_slim(@test ∂out[3] isa Tuple)
println_slim(@test ∂out[3][1] ≈ gP_zy)
println_slim(@test ∂out[5] ≈ gW_zy)

##

@info("  forward-mode pfwd_ed (jacobian pushforward)")

dP = randn(nX, in_dim)
(Bf, dBf), _ = ET.pfwd_ed(l, (P, dP, X), ps, st)
println_slim(@test Bf ≈ Bman)
println_slim(@test all(dBf[i, :] ≈ ps.W[:, :, X[i]] * dP[i, :] for i = 1:nX))
