#
# CP / TRACE format — end-to-end pipeline test.
#
# Drives the format through the real ET pipeline
#   graph → EdgeEmbed(Rnl/Ylm) → pool (PooledSparseProduct) → CPACElayer → energy
# and checks the position gradient (forces) and parameter gradient against
# ForwardDiff / finite differences. This validates that the CP rrules compose
# with the embedding + pooling rrules through Zygote.
#

using EquivariantTensors, StaticArrays, Random, LinearAlgebra, Test
using LuxCore, DecoratedParticles, Zygote, ForwardDiff
using DecoratedParticles: PState
using ACEbase.Testing: println_slim
import EquivariantTensors as ET
import Polynomials4ML as P4ML
import SpheriCart

rng = MersenneTwister(11)

@info("CP/TRACE: end-to-end pipeline (forces + parameter gradients)")

Dtot = 8; maxl = 4; ORD = 3; K = 5
rbasis = P4ML.ChebBasis(Dtot+1)
ybasis = SpheriCart.SolidHarmonics(maxl; static=true)
mb = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
        level = bb -> sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)
cpbasis = ET.cp_equivariant_tensor(; LL=(0,), mb_spec=mb,
        Rnl_spec=P4ML.natural_indices(rbasis),
        Ylm_spec=P4ML.natural_indices(ybasis), basis=real, rank=K)
cplayer = ET.CPACElayer(cpbasis, (1,))
rembed = ET.EdgeEmbed(ET.StateEmbed(ET.state_transform(x -> 1/(1+norm(x.𝐫)^2)), rbasis))
yembed = ET.EdgeEmbed(ET.StateEmbed(ET.state_transform(x -> x.𝐫), ybasis))

psr, str = LuxCore.setup(rng, rembed)
psy, sty = LuxCore.setup(rng, yembed)
psc, stc = LuxCore.setup(rng, cplayer)

X = ET.Testing.rand_graph(20; nneigrg = 4:8)

function energy(G, psc)
   Rnl, _ = rembed(G, psr, str)
   Ylm, _ = yembed(G, psy, sty)
   A = ET.ka_evaluate(cpbasis.abasis, (Rnl, Ylm))
   F, _ = cplayer(A, psc, stc)
   return sum(F[1])
end

# ∂energy/∂positions: Zygote vs ForwardDiff (over the edge positions)
∇X_zy = Zygote.gradient(G -> energy(G, psc), X)[1]
function grad_X_fd(G)
   _replace(Rmat) = ET.ETGraph(G.ii, G.jj, G.first, G.node_data,
         [ PState(𝐫 = SVector{3}(Rmat[:, i])) for i in 1:size(Rmat, 2) ],
         G.graph_data, G.maxneigs)
   Rmat = reinterpret(reshape, Float64, [ x.𝐫 for x in G.edge_data ])
   ∇ = ForwardDiff.gradient(R -> energy(_replace(R), psc), Rmat)
   return [ SVector{3}(∇[:, i]) for i in 1:size(∇, 2) ]
end
∇X_fd = grad_X_fd(X)
∇zy_𝐫 = [ x.𝐫 for x in ∇X_zy.edge_data ]
println_slim(@test ∇zy_𝐫 ≈ ∇X_fd)

# ∂energy/∂λ: Zygote vs finite differences
gλ = Zygote.gradient(p -> energy(X, p), psc)[1]
λ = psc.λ[1]; gλ_fd = zero(λ); h = 1e-6
for idx in eachindex(λ)
   λp = copy(λ); λp[idx] += h; λm = copy(λ); λm[idx] -= h
   gλ_fd[idx] = (energy(X, (basis=psc.basis, λ=(λp,))) -
                 energy(X, (basis=psc.basis, λ=(λm,)))) / (2h)
end
println_slim(@test gλ.λ[1] ≈ gλ_fd)
