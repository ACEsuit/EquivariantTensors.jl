

using EquivariantTensors, StaticArrays, Test, ForwardDiff, 
      DecoratedParticles, Zygote, LinearAlgebra

using ACEbase.Testing: println_slim, print_tf 

import EquivariantTensors as ET
import DecoratedParticles as DP

##


rand_x() = PState(q = randn(), r = randn(SVector{3, Float64}), z = rand(1:10))

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


function grad_fd2(f, x, args...)
   v = VState(x) 
   x_nt = getfield(x, :x)
   v_nt = _ctsnt(x_nt)  # extract continuous variables into an SVector 
   v = _nt2svec(v_nt)
   _fvec = _v -> f(STATE(_replace(x_nt, _svec2nt(_v, v_nt))), args...)
   g = ForwardDiff.gradient(_fvec, _nt2svec(v_nt))
   return VState(_svec2nt(g, v_nt))  # return as NamedTuple
end 

v = VState(x) 
TV = typeof(v) 
sv = reinterpret(SVector{4, Float64}, v) 
reinterpret(TV, sv)
_f(_sv) = f(x + reinterpret(TV, _sv))


##

f = F(@SVector randn(10))
x = rand_x() 

f(x)
g0 = grad_man(f, x)
g1 = grad_1(f, x)
g2 = grad_zy(f, x)
g3 = ET.DiffNT.grad_fd(f, x)

g0 ≈ g1 ≈ g2 ≈ g3

##

using BenchmarkTools

@btime grad_man($f, $x)
@btime grad_1($f, $x)
@btime grad_zy($f, $x)
@btime ET.DiffNT.grad_fd($f, $x)

## 

# Differentiate a composition of a transform and a basis 

using Polynomials4ML
import Polynomials4ML as P4ML
using Lux, LuxCore, Random 
rng = MersenneTwister(1234)

function _pb(trans, X, dP) 
   _pb1(x, dp) = ET.DiffNT.grad_fd(_x -> dot(trans(_x), dp), x) 
   return broadcast(_pb1, X, dP)
end


basis = ChebBasis(10)
trans = x -> 1 / (1 + sum(abs2, x.r))
embed = ET.EdgeEmbedDP(WrappedFunction(trans), basis; name = "Rnl")
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x)
X = G.edge_data

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
dY = ET.DiffNT.grad_fd.(Ref(trans), X)
∂P1 = dY .* dP 
∂P2 = _pb(trans, X, dP)

(_P3, _∂P3), _ = ET.evaluate_ed(embed, G, ps, st)
P3 = ET.rev_reshape_embedding(_P3, G)
∂P3 = ET.rev_reshape_embedding(_∂P3, G)

P ≈ P3
all(∂P1 .≈ ∂P2 .≈ ∂P3)

## 

basis = real_solidharmonics(4)
trans = x -> x.r 
embed = ET.EdgeEmbedDP(WrappedFunction(trans), basis; name = "Ylm")
ps, st = LuxCore.setup(rng, embed)

G = ET.Testing.rand_graph(20; randedge = rand_x)
X = G.edge_data

Y = trans.(X)
P, dP = P4ML.evaluate_ed(basis, Y)
∂P1 = map(dr -> VState(q = 0.0, r = dr), dP)
∂P2 = _pb(trans, X, dP)

(_P3, _∂P3), _ = ET.evaluate_ed(embed, G, ps, st)
P3 = ET.rev_reshape_embedding(_P3, G)
∂P3 = ET.rev_reshape_embedding(_∂P3, G)

P ≈ P3
all(∂P1 .≈ ∂P2 .≈ ∂P3)

