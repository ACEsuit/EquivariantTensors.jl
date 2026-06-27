

using EquivariantTensors, StaticArrays, Test,
      DecoratedParticles, LinearAlgebra,
      Polynomials4ML, Lux, LuxCore, Random

using ACEbase.Testing: println_slim, print_tf

import EquivariantTensors as ET
import DecoratedParticles as DP
import Polynomials4ML as P4ML
import SpheriCart

rng = MersenneTwister(1234)

##

@info("Tests of DecoratedParticles usage")

# NOTE: correctness of grad_fd itself (vs manual / componentwise FD
#       gradients) is tested in DecoratedParticles,
#       test/test_differentiation.jl; here we only test the ET
#       embedding layers built on top of it.

# generate a random DP
rand_x_dp() = PState(q = randn(), r = randn(SVector{3, Float64}), z = rand(1:10))

##

@info("Test StateEmbed - Radial Basis")

basis = ChebBasis(10)
trans = x -> 1 / (1 + sum(abs2, x.r))
embed = ET.StateEmbed(ET.state_transform(trans), basis)
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x_dp)
X = G.edge_data

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
dY = DP.grad_fd.(Ref(trans), X)
∂P1 = dY .* dP

P2a, _ = embed(X, ps, st)
(P2, ∂P2), _ = ET.evaluate_ed(embed, X, ps, st)

println_slim(@test P ≈ P2 ≈ P2a)
println_slim(@test all(∂P1 .≈ ∂P2 ))

## 

@info("Test StateEmbed - Solid Harmonics Basis")

basis = SpheriCart.SolidHarmonics(4)
trans = x -> x.r 
embed = ET.StateEmbed(ET.state_transform(trans), basis)
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x_dp)
X = G.edge_data

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
∂P1 = map(dr -> VState(q = 0.0, r = dr), dP)

(P2, ∂P2), _ = ET.evaluate_ed(embed, X, ps, st)

println_slim(@test P ≈ P2)
println_slim(@test all(∂P1 .≈ ∂P2))

