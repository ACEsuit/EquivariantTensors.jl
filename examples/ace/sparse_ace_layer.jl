import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, Combinatorics, LinearAlgebra, Random
using Zygote 
using LuxCore

Dtot = 10   # total degree; specifies the trunction of embeddings and correlations
maxl = 8    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

##

# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; L = 2, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

##
𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = (0, 2), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real)

# Construct the ACE layer model with specified output feature sizes for each L
l = ET.sparse_ace_layer(rbasis, ybasis, 𝔹basis, Dict(0 => 5, 2 => 2))
ps, st = LuxCore.setup(MersenneTwister(1234), l)

rand_sphere() = ( u = randn(SVector{3, Float64}); u / norm(u) )
rand_x() = (0.1 + 0.9 * rand()) * rand_sphere()

nX = 7
𝐫 = [ rand_x() for _ = 1:nX ]

# Evaluate the ACE layer on the generated configuration
φ = l(𝐫, ps, st)[1]
typeof(φ) # Tuple{Vector{Float64}, Vector{SVector{5, Float64}}}

# Additional packages for optimization and testing
using Optimisers: destructure          
using Test                           
using ACEbase.Testing: print_tf, fdtest

# --- Finite Difference Test: wrt input configuration 𝐫 ---
p, s = destructure(𝐫)         
bu = rand(length(p))  
_BB(t) = p + t * bu 
f = φ -> destructure(φ)[1]

val = f(l(𝐫, ps, st)[1])
u = randn(size(val))

F1(t) = dot(u, f(l(s(_BB(t)), ps, st)[1]))
dF(t) = Zygote.gradient(t -> F1(t), t)[1]

# Compare gradient from AD with finite difference approximation
print_tf(@test fdtest(F1, dF, 0.0; verbose=true))

# --- Finite Difference Test: wrt model parameters `ps` ---

p, s = destructure(ps)
bu = rand(length(p))
_BB(t) = p + t * bu
F1(t) = dot(u, f(l(𝐫, s(_BB(t)), st)[1]))
dF(t) = Zygote.gradient(t -> F1(t), t)[1]

print_tf(@test fdtest(F1, dF, 0.0; verbose=true))