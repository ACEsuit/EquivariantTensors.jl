

using EquivariantTensors, StaticArrays, Test, ForwardDiff, 
      DecoratedParticles, Zygote, LinearAlgebra, 
      Polynomials4ML, Lux, LuxCore, Random 

using ACEbase.Testing: println_slim, print_tf 

import EquivariantTensors as ET
import DecoratedParticles as DP
import Polynomials4ML as P4ML

rng = MersenneTwister(1234)

##

@info("Tests of DecoratedParticles usage") 

# generate a random DP 
rand_x_dp() = PState(q = randn(), r = randn(SVector{3, Float64}), z = rand(1:10))

module TestDP 
   using StaticArrays, ForwardDiff, Zygote
   import DecoratedParticles: PState, VState
   import EquivariantTensors as ET

   # random expression, but representative in terms of simplicity 
   struct F{N, T}; W::SVector{N, T}; end
   evaluate(f::F, x) = sum(x.r .* x.r) * x.q / (1 + f.W[x.z]^2)   
   (f::F)(x::PState) = evaluate(f, x)

   # manual gradient 
   function grad_man(f::F, x) 
      r2 = sum(x.r .* x.r)
      w = 1 / (1 + f.W[x.z]^2)
      return VState(q = r2 * w, r = 2 * x.r * x.q * w, ) 
   end

   # gradient via ForwardDiff                       
   function grad_1(f::F, x) 
      ∂q = ForwardDiff.derivative(q -> evaluate(f, PState(q=q,   r=x.r, z=x.z)), x.q)
      ∂r = ForwardDiff.gradient(r -> evaluate(f, PState(q=x.q, r=r,   z=x.z)), x.r)
      return VState(q = ∂q, r = ∂r)
   end

   function grad_zy(f::F, x) 
      g = Zygote.gradient(evaluate, f, x)
      return g[2]
   end
end 

##

@info("Test diff of scalar fcn w.r.t. a DP")

f = TestDP.F(@SVector randn(10))
x = rand_x_dp() 

f(x)
g0 = TestDP.grad_man(f, x)
g1 = TestDP.grad_1(f, x)
g2 = TestDP.grad_zy(f, x)
g3 = ET.DiffNT.grad_fd(f, x)

println_slim(@test g0 ≈ g1 ≈ g2 ≈ g3)

##

# performance of grad_fd is not ideal, but may still be sufficient 
# we will need to see how this behaves in larger tests. 

# using BenchmarkTools
# @btime grad_man($f, $x)    # 3.3ns 
# @btime grad_1($f, $x)      # 6.5ns
# @btime grad_zy($f, $x)     # 1.4us 
# @btime ET.DiffNT.grad_fd($f, $x)   # 8.3ns

## 

@info("Test EmbedDP - Radial Basis") 

basis = ChebBasis(10)
trans = x -> 1 / (1 + sum(abs2, x.r))
embed = ET.EmbedDP(ET.dp_transform(trans), basis)
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x_dp)
X = G.edge_data
X_nt = [ getfield(x, :x) for x in X ]

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
dY = ET.DiffNT.grad_fd.(Ref(trans), X)
∂P1 = dY .* dP 

P2a, _ = embed(X, ps, st)
(P2, ∂P2), _ = ET.evaluate_ed(embed, X, ps, st)

# TODO: this test is temporarily broken because the transform pullback is 
# a bit hacky due to GPU requirements. 
# (P3, _∂P3), _ = ET.evaluate_ed(embed, X_nt, ps, st)
# ∂P3 = VState.(_∂P3) 

println_slim(@test P ≈ P2 ≈ P2a)   # ≈ P3
println_slim(@test all(∂P1 .≈ ∂P2 ))  # .≈ ∂P3

## 

@info("Test EdgeEmbedDP - Solid Harmonics Basis") 

basis = real_solidharmonics(4)
trans = x -> x.r 
embed = ET.EmbedDP(ET.dp_transform(trans), basis)
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x_dp)
X = G.edge_data
X_nt = [ getfield(x, :x) for x in X ]

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
∂P1 = map(dr -> VState(q = 0.0, r = dr), dP)

(P2, ∂P2), _ = ET.evaluate_ed(embed, X, ps, st)
(P3, _∂P3), _ = ET.evaluate_ed(embed, X_nt, ps, st)
∂P3 = VState.(_∂P3)

println_slim(@test P ≈ P2 ≈ P3)
println_slim(@test all(∂P1 .≈ ∂P2 .≈ ∂P3))

