

using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "Polynomials4ML.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "DecoratedParticles"))

##

using StaticArrays, Random, LuxCore, Test, LinearAlgebra, ForwardDiff
using EquivariantTensors

import EquivariantTensors as ET
import Polynomials4ML as P4ML 
import DecoratedParticles as DP

using ACEbase.Testing: print_tf, println_slim

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##
#
# Generate an radial embedding with selection of splines 

NCAT = 4 
indim = 20; outdim = 10 
trans_states = (; params = rand(NCAT))
trans = ET.dp_transform( (x, st) -> 1 - 1 / (0.5 + st.params[x.c] * norm(x.𝐫)), 
                         trans_states )
polys = P4ML.ChebBasis(indim)
sellin = ET.SelectLinL(length(polys), outdim, NCAT, x -> x.c)
rembed = ET.EmbedDP(trans, polys, sellin)

ps, st = LuxCore.setup(rng, rembed)

for c = 1:NCAT
   nfeat = size(ps.post.W, 2)
   ps.post.W[:, :, c] = ps.post.W[:, :, c] .* ((1:nfeat).^(-2))'
end

## 
#
# splinify the embedding 

Nspl = 100 
WW = ps.post.W 
splines = [ 
      P4ML.splinify( y -> WW[:, :, i] * polys(y), -1.0, 1.0, Nspl ) 
      for i in 1:size(WW, 3)  ]
states = [ P4ML._init_luxstate(spl) for spl in splines ]
spl = ET.TransSelSplines(trans, nothing, sellin.selector, splines[1], states)

# I want it to look something like this: 
# spl_100 = ET.transsel_splines(trans, splines, nothing) 

ps_spl, st_spl = LuxCore.setup(rng, spl)

## 

rand_X() = [ DP.PState( 𝐫 = (@SVector randn(3)), c = rand(1:NCAT) )
             for _ = 1:rand(100:300) ]

##

Random.seed!(1234)  # new seed to make sure the tests are ok.
for ntest = 1:30 
   X = rand_X() 
   P1, _ = rembed(X, ps, st)
   P2, _ = spl(X, ps_spl, st_spl)
   (P1a, dP1a), _ = ET.evaluate_ed(rembed, X, ps, st)
   (P2a, dP2a), _ = ET.evaluate_ed(spl, X, ps_spl, st_spl)
   print_tf(@test P2a ≈ P2)
   print_tf(@test norm(P1 - P2, Inf) < 1e-5)
   print_tf(@test maximum(norm.(dP1a - dP2a)) < 1e-3)
end

##

#=

@info("Check GPU evaluation") 
using Metal 
dev = Metal.mtl
ps_32 = ET.float32(ps_spl)
st_32 = ET.float32(st_spl)
ps_dev = dev(ps_32)
st_dev = dev(st_32)

X = rand_X() 
X_32 = ET.float32.(X)
X_dev = dev(X_32)

=#