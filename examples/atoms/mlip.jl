#
# This example is incomplete since it doesn't yet allow the computation 
# of gradients i.e. forces. 
#

include(@__DIR__() * "/atomsext.jl")

## 

@info("Convert a structure to an ETGraph")

using AtomsBuilder, Unitful 

sys = rattle!(bulk(:Si, cubic=true) * (3,3,2), 0.1u"Å")
rcut = 5.0u"Å"
G_sys = ETAtomsExt.interaction_graph(sys, rcut)

## 

@info("Build a simple ACE model")

using Lux, Random, LinearAlgebra  
import EquivariantTensors as ET
import Polynomials4ML as P4ML

# generate a model 
Dtot = 9     # total degree; specifies the trunction of embeddings and correlations
maxl = 5     # maximum degree of spherical harmonics 
ORD = 3      # correlation-order (body-order = ORD + 1)

# To test with a larger model replace with the following 
# Dtot = 16; maxl = 10; ORD = 3

polys = P4ML.ChebBasis(Dtot+1)
Rnl_spec = P4ML.natural_indices(polys)
ybas = P4ML.real_sphericalharmonics(maxl; T = Float32, static=true)
Ylm_spec = P4ML.natural_indices(ybas)

# generate the embedding layer 
rcut_u = ustrip(rcut)
ycut = 1 / 2  # 1 / (1 + r / rcut) is the distrance transform 
env = y -> (y - ycut)^2 * (y + ycut)^2

rbasis = Chain( y = ET.NTtransform(x -> 1 / (1+norm(x.𝐫/rcut_u))),
                P = SkipConnection(
                    polys,
                    WrappedFunction( PY -> env.(PY[2]) .* PY[1] )
                ) )

ybasis = Chain( trans = ET.NTtransform(x -> x.𝐫), 
                basis = ybas )

# rbasis = ET.TransformedBasis( ET.NTtransform(x -> 1 / (1+norm(x.𝐫/rcut_u))), 
#                               P4ML.ChebBasis(Dtot+1), 
#                               ET.Envelope( (x, y) -> env(y) ) )
# ybasis = ET.TransformedBasis( ET.NTtransform(x -> x.𝐫), 
#                               P4ML.real_sphericalharmonics(maxl; T = Float32, static=true) )

embed = ET.EdgeEmbed( BranchLayer(; Rnl = rbasis, Ylm = ybasis) )

acel = let ORD = ORD, Dtot = Dtot, maxl = maxl 
   mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
   𝔹basis = ET.sparse_equivariant_tensor(; 
                  L = 0,    # this says to produce on an invariant basis output
                  mb_spec = mb_spec, 
                  Rnl_spec = Rnl_spec, 
                  Ylm_spec = Ylm_spec, 
                  basis = real )

   acel = ET.SparseACElayer(𝔹basis, (1,))  # the (1,) says just one output channel 
end

# build the model from the two layers
model = Lux.Chain(; embed = embed,  # embedding layer 
                    ace = acel,     # ACE layer / correlation layer 
                    energy = WrappedFunction(x -> sum(x[1]))   # sum up to get a total energy 
                  )
ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)

E1, _ = model(G_sys, ps, st) # evaluate the model on the graph

##
# Differentiation 
# crazy stuff - this seems to "just work" with Zygote

using Zygote 

∇E = Zygote.gradient(G -> model(G, ps, st)[1], G_sys)[1]

