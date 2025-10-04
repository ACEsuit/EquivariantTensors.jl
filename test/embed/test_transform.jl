
using EquivariantTensors, StaticArrays, Test, Polynomials4ML, LuxCore, Random 
using ACEbase.Testing: println_slim, print_tf 
using ACEbase: evaluate, evaluate_ed 

rng = Random.default_rng(1234)

import ACEbase 
import EquivariantTensors as ET
import Polynomials4ML as P4ML


@info("Testing transformed basis ")

##

basis = ChebBasis(5)

##

@info("   Scalar input, no transforms")
tbasis = ET.TransformedBasis(;basis = basis)
X = [ 2 * rand() - 1 for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, X)
println_slim(@test P1 ≈ P2) 

##

@info("   NT input, transformed into scalar transform")
transin = ET.NTtransform(x -> 1 / (1+x.r))
tbasis = ET.TransformedBasis(; transin = transin, 
                                 basis = basis)
X = [ (;r = 10 * rand()) for _ in 1:10 ]
ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)
P2 = evaluate(tbasis.basis, transin.(X))
println_slim(@test P1 ≈ P2) 

##

@info("   NT input, scalar input transform, select output transform")
transin = ET.NTtransform(x -> 1 / (1+x.r))

struct SelectLinear <: AbstractLuxLayer
   W::Array{Float64, 3} 
end

LuxCore.initialparameters(rng::AbstractRNG, l::SelectLinear) = (W = l.W,)
LuxCore.initialstates(rng::AbstractRNG, l::SelectLinear) = NamedTuple()

function EquivariantTensors.evaluate(l::SelectLinear, P, X, _Y, ps, st) 
   B = zeros(Float64, length(X), size(ps.W, 2))
   for i in 1:length(X)
      B[i, :] = P[SA[i], :] * ps.W[:, :, X[i].z]
   end
   return B
end

transout = SelectLinear(randn(length(basis), 4, 3))

tbasis = ET.TransformedBasis(; transin = transin, 
                               basis = basis, 
                               transout = transout) 

X = [ (;r = 10 * rand(), z = rand(1:3)) for _ in 1:10 ]

ps, st = LuxCore.setup(rng, tbasis)

P1, _ = tbasis(X, ps, st)

Y = transin.(X)
T2 = evaluate(tbasis.basis, Y)
B2 = reduce(vcat, T2[i, :]' * transout.W[:, :, X[i].z] for i = 1:length(X))
println_slim(@test P1 ≈ B2)
