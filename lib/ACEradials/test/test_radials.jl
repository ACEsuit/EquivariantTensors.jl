
using ACEradials, StaticArrays, Random, Test
using ACEradials: evaluate, evaluate_ed
import Zygote
const R = ACEradials

##

@info("Test learnable / splined radial basis")

elements = (8, 1)   # O, H atomic numbers
NZ = length(elements)
rin0cut = (rin = 0.0, r0 = 1.0, rcut = 5.0)
rin0cuts = SMatrix{NZ, NZ}([rin0cut for i = 1:NZ, j = 1:NZ])
spec = [ (n = n, l = l) for n = 1:5, l = 0:2 ][:]

basis = R.learnable_Rnl_basis(elements, rin0cuts; spec = spec, maxq = 6)

@test length(basis) == length(spec)
@test R.parameterlength(basis) == length(spec) * 6 * NZ^2

rng = MersenneTwister(1234)
ps = R.initialparameters(rng, basis)
st = R.initialstates(rng, basis)

spl = R.splinify(basis, ps; nnodes = 300)
@test spl isa R.SplineRnlBasis
@test length(spl) == length(basis)
@test R.splinify(spl, ps) === spl   # splinify of a spline is a no-op
# splinify must not mutate the meta of the input basis
@test !haskey(basis.meta, "info")

##

@info("   learnable vs splined values + derivatives agree to spline tol")
rs = range(0.2, 4.8, length = 50)
errs  = Float64[]
derrs = Float64[]
for r in rs, (zi, zj) in [(8,8), (8,1), (1,8), (1,1)]
   Rl = evaluate(basis, r, zi, zj, ps, st)
   Rs = evaluate(spl, r, zi, zj, ps, st)
   push!(errs, maximum(abs.(Rl .- Rs)))
   _, Rl_d = evaluate_ed(basis, r, zi, zj, ps, st)
   _, Rs_d = evaluate_ed(spl, r, zi, zj, ps, st)
   push!(derrs, maximum(abs.(Rl_d .- Rs_d)))
end
@test maximum(errs)  < 1e-5
@test maximum(derrs) < 1e-4

##

@info("   finite-difference check of evaluate_ed")
for basis_ in (basis, spl), r in (0.7, 2.3, 4.1)
   h = 1e-6
   _, Rd = evaluate_ed(basis_, r, 8, 1, ps, st)
   Rp = evaluate(basis_, r + h, 8, 1, ps, st)
   Rm = evaluate(basis_, r - h, 8, 1, ps, st)
   @test maximum(abs.(Rd .- (Rp .- Rm) ./ (2h))) < 1e-5
end

##

@info("   batched evaluation matches the scalar interface")
rs_b = collect(range(0.5, 4.5, length = 8))
zjs  = rand(rng, (1, 8), 8)
for basis_ in (basis, spl)
   Rb = R.evaluate_batched(basis_, rs_b, 8, zjs, ps, st)
   @test size(Rb) == (length(rs_b), length(basis_))
   for j = 1:length(rs_b)
      @test Rb[j, :] ≈ evaluate(basis_, rs_b[j], 8, zjs[j], ps, st)
   end
   Rb_v, Rb_d = R.evaluate_ed_batched(basis_, rs_b, 8, zjs, ps, st)
   @test Rb_v ≈ Rb
end

##

@info("   Zygote gradient w.r.t. Wnlq matches finite differences")
W0 = ps.Wnlq
loss = W -> sum(abs2,
         R.evaluate_batched(basis, rs_b, 8, zjs, (Wnlq = W,), st))
gz = Zygote.gradient(loss, W0)[1]
@test size(gz) == size(W0)
h = 1e-6
for _ = 1:20
   i = rand(rng, eachindex(W0))
   Wp = copy(W0); Wp[i] += h
   Wm = copy(W0); Wm[i] -= h
   gfd = (loss(Wp) - loss(Wm)) / (2h)
   @test abs(gz[i] - gfd) < 1e-5 * max(1.0, abs(gfd))
end

# splined basis has no parameters; the rrule returns no tangents but
# must not error
gspl = Zygote.gradient(
         W -> sum(R.evaluate_batched(spl, rs_b, 8, zjs, (Wnlq = W,), st)),
         W0)[1]
@test isnothing(gspl)

##

@info("   envelope evaluate_d partial derivatives")
env2 = basis.envelopes[1, 1]
@test env2 isa R.PolyEnvelope2sX
let r = 2.3, x = 0.4, h = 1e-6
   dr, dx = R.evaluate_d(env2, r, x)
   @test dr == 0
   @test abs(dx - (evaluate(env2, r, x + h) - evaluate(env2, r, x - h)) / (2h)) < 1e-6
end
env1 = R.PolyEnvelope1sR(5.0, 1)
let r = 2.3, x = 0.4, h = 1e-6
   dr, dx = R.evaluate_d(env1, r, x)
   @test dx == 0
   @test abs(dr - (evaluate(env1, r + h, x) - evaluate(env1, r - h, x)) / (2h)) < 1e-6
end
