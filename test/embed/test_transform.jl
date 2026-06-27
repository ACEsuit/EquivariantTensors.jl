
using EquivariantTensors, StaticArrays, Test, Polynomials4ML, LuxCore, Random 
using ACEbase.Testing: println_slim, print_tf 
using ACEbase: evaluate, evaluate_ed
using DecoratedParticles
using LinearAlgebra: dot
import Zygote, ForwardDiff

rng = Random.default_rng(1234)

import ACEbase 
import EquivariantTensors as ET
import Polynomials4ML as P4ML


@info("Testing transformed basis ")

##

basis = ChebBasis(5)

##

@info("  DP input, transformed into scalar transform")
trans = ET.state_transform(x -> 1 / (1+x.r))
tbasis = ET.StateEmbed(trans, basis)
X = [ PState(;r = 10 * rand()) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, trans(X, st.trans))
println_slim(@test P1 ≈ P2) 

##

@info("   DP input, scalar input transform, select output transform")

trans = ET.state_transform(x -> 1 / (1+x.r))
basis = ChebBasis(5)
sellin = ET.SelectLinL(5, 10, 3, x -> x.z)
tbasis = ET.StateEmbed(trans, basis, sellin)
ps, st = LuxCore.setup(rng, tbasis)

X = [ PState(;r = 10 * rand(), z = rand(1:3)) for _ in 1:10 ]

P1, _ = tbasis(X, ps, st)

Y = trans(X, st.trans)
T2 = evaluate(basis, Y)
B2 = reduce(vcat, T2[i, :]' * ps.post.W[:, :, X[i].z]'
                  for i = 1:length(X))
println_slim(@test P1 ≈ B2)

##

@info("   StateEmbed + SelectLinL: gradients & jacobian (vector positions)")

# salvaged from the removed dormant/test_embed.jl (old NTtransform API): the
# only test exercising SelectLinL *through* StateEmbed, i.e.
# StateEmbed.evaluate_ed with a non-IDpost post. (StateEmbed's rrule returns
# NoTangent for ps, so only the input/position gradient flows here; SelectLinL
# param grads are covered in test/utils/test_selectlinl.jl.)

npoly, nout, ncat = 8, 6, 3
embed = ET.StateEmbed(ET.state_transform(x -> 1 / (1 + sum(abs2, x.r))),
                      ChebBasis(npoly),
                      ET.SelectLinL(npoly, nout, ncat, x -> x.z))
ps, st = LuxCore.setup(rng, embed)
X = [ PState(r = randn(SVector{3, Float64}), z = rand(1:ncat)) for _ = 1:12 ]
nX = length(X)

# forward vs manual (z-selected linear combination of the Cheb basis)
B1, _ = embed(X, ps, st)
Yr = [ 1 / (1 + sum(abs2, x.r)) for x in X ]
T = ChebBasis(npoly)(Yr)
B1man = reduce(vcat, (ps.post.W[:, :, X[i].z] * T[i, :])' for i = 1:nX)
println_slim(@test B1 ≈ B1man)

# ForwardDiff position gradient of dot(Δ, embed) — the FD reference
Δ = randn(nX, nout)
Rmat = reduce(hcat, [ Vector(x.r) for x in X ])      # 3 × nX
function _loss(Rm)
   Xn = [ PState(r = SVector{3}(Rm[:, i]), z = X[i].z) for i = 1:nX ]
   return dot(Δ, embed(Xn, ps, st)[1])
end
gfd = ForwardDiff.gradient(_loss, Rmat)
g_fd = [ SVector{3}(gfd[:, i]) for i = 1:nX ]

# jacobian via evaluate_ed: contract ∂P with Δ → position gradient (pfwd_ed path)
(Pe, ∂Pe), _ = ET.evaluate_ed(embed, X, ps, st)
println_slim(@test Pe ≈ B1)
g_ed = [ sum(Δ[i, j] * ∂Pe[i, j].r for j = 1:nout) for i = 1:nX ]
println_slim(@test all(g_ed[i] ≈ g_fd[i] for i = 1:nX))

# reverse-mode input gradient via the StateEmbed ∘ SelectLinL rrule
g_zy = Zygote.gradient(_X -> dot(Δ, embed(_X, ps, st)[1]), X)[1]
println_slim(@test all(g_zy[i].r ≈ g_fd[i] for i = 1:nX))
