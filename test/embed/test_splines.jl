

using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();

##

using StaticArrays, Random, LuxCore, Test, LinearAlgebra, ForwardDiff, 
      EquivariantTensors, Lux

import EquivariantTensors as ET
import Polynomials4ML as P4ML 
import DecoratedParticles as DP

using ACEbase.Testing: print_tf, println_slim

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##
#
# Generate an radial embedding with envelope and selection of 
# linear transform
# this is a typical radial embedding construction as used in ACEpotentials 
# and ACEhamiltonians 

NCAT = 4 
indim = 20; outdim = 10 
trans_states = (; params = rand(NCAT))
trans_fun = let 
   (x, st) -> 1 - 2 / (1 + st.params[x.c] * norm(x.𝐫))
end 
env_fun = let 
   y -> (1 - y^2)^2 
end
trans = ET.dp_transform(trans_fun, trans_states )
polys_y = P4ML.ChebBasis(indim)
Penv = P4ML.wrapped_basis( BranchLayer(
         polys_y,   # y -> P
         WrappedFunction( y -> env_fun.(y) ),  # y -> fₑₙᵥ
         fusion = WrappedFunction( Pe -> Pe[2] .* Pe[1] )  
      ) ) 
sel_fun = let 
   x -> x.c 
end 
sellin = ET.SelectLinL(length(polys_y), outdim, NCAT, x -> x.c)
rembed = ET.EmbedDP(trans, Penv, sellin)
ps, st = LuxCore.setup(rng, rembed)

# smoothen the splines so that we can sensible errors with few spline points 
for c = 1:NCAT
   nfeat = size(ps.post.W, 2)
   ps.post.W[:, :, c] = ps.post.W[:, :, c] .* ((1:nfeat).^(-3))'
end

## 
#
# splinify the embedding 

# could try false for local testing, but CI should use true 
# which is the more interesting scenario for most applications 
extract_envelope = true
spl_30 = ET.trans_splines(rembed, ps, st; 
                           yrange = (-1.0, 1.0), nspl = 30, 
                           extract_envelope = extract_envelope)
spl_100 = ET.trans_splines(rembed, ps, st; 
                           yrange = (-1.0, 1.0), nspl = 100,
                           extract_envelope = extract_envelope)

ps_30, st_30 = LuxCore.setup(rng, spl_30)
ps_100, st_100 = LuxCore.setup(rng, spl_100)

## 

rand_X() = [ DP.PState( 𝐫 = (@SVector randn(3)), c = rand(1:NCAT) )
             for _ = 1:rand(100:300) ]

##

Random.seed!(1)  # new seed to make sure the tests are ok.
for ntest = 1:30 
   local X, P1 
   X = rand_X() 
   P1, _ = rembed(X, ps, st)
   P_30, _ = spl_30(X, ps_30, st_30)
   P_100, _ = spl_100(X, ps_100, st_100)

   (P1a, dP1a), _ = ET.evaluate_ed(rembed, X, ps, st)
   (P_30a, dP_30a), _ = ET.evaluate_ed(spl_30, X, ps_30, st_30)
   (P_100a, dP_100a), _ = ET.evaluate_ed(spl_100, X, ps_100, st_100)

   print_tf(@test norm(P1 - P_30, Inf) < 1e-3)
   print_tf(@test norm(P1 - P_100, Inf) < 1e-5)
   print_tf(@test P_30a ≈ P_30)
   print_tf(@test P_100a ≈ P_100)
   print_tf(@test maximum(norm.(dP1a - dP_30a)) < 3e-2)
   print_tf(@test maximum(norm.(dP1a - dP_100a)) < 2e-3)
end
println() 

##


@info("Check GPU evaluation") 
using Metal 
dev = Metal.mtl
ps_32 = ET.float32(ps_100)
st_32 = ET.float32(st_100)
ps_dev = dev(ps_32)
st_dev = dev(st_32)

X = rand_X() 
X_32 = ET.float32.(X)
X_dev = dev(X_32)

P1, _ = spl_100(X_32, ps_32, st_32)
P2_dev, _ = spl_100(X_dev, ps_dev, st_dev)
P2 = Array(P2_dev)
println_slim(@test P1 ≈ P2)

(P1, ∂P1), _ = ET.evaluate_ed(spl_100, X_32, ps_32, st_32)
(P2_dev, ∂P2_dev), _ = ET.evaluate_ed(spl_100, X_dev, ps_dev, st_dev)
P2 = Array(P2_dev)
∂P2 = Array(∂P2_dev)
println_slim(@test P1 ≈ P2 )
println_slim(@test all(norm.(∂P1 .- ∂P2) .< 1e-5))
