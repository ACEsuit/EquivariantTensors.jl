
using EquivariantTensors, StaticArrays, Test, Polynomials4ML, LuxCore, Random 
using ACEbase.Testing: println_slim, print_tf 
using ACEbase: evaluate, evaluate_ed 
using DecoratedParticles

rng = Random.default_rng(1234)

import ACEbase 
import EquivariantTensors as ET
import Polynomials4ML as P4ML


@info("Testing transformed basis ")

##

basis = ChebBasis(5)

##

@info("  DP input, transformed into scalar transform")
trans = ET.dp_transform(x -> 1 / (1+x.r))
tbasis = ET.EmbedDP(trans, basis)
X = [ PState(;r = 10 * rand()) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, trans(X, st.trans))
println_slim(@test P1 ≈ P2) 

##

@info("   DP input, scalar input transform, select output transform")

trans = ET.dp_transform(x -> 1 / (1+x.r))
basis = ChebBasis(5)
sellin = ET.SelectLinL(5, 10, 3, x -> x.z)
tbasis = ET.EmbedDP(trans, basis, sellin)
ps, st = LuxCore.setup(rng, tbasis)

X = [ PState(;r = 10 * rand(), z = rand(1:3)) for _ in 1:10 ]

P1, _ = tbasis(X, ps, st)

Y = trans(X, st.trans)
T2 = evaluate(basis, Y)
B2 = reduce(vcat, T2[i, :]' * ps.post.W[:, :, X[i].z]' 
                  for i = 1:length(X))
println_slim(@test P1 ≈ B2)
