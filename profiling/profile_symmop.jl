using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
import Polynomials4ML as P4ML 
import EquivariantTensors as ET
using StaticArrays, SparseArrays, LinearAlgebra, Zygote, Random

##

# Some model parameters that we will use: 
Dtot = 18   # total degree; specifies the trunction of embeddings and correlations
maxl = 12    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

# first specify the radial and angular embeddings 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
ybasis = P4ML.complex_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

# generate the nnll basis pre-specification
nnll_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

##

# in the pre-specification we only imposed the total degree truncation, everything 
# else will be handled by the symmetrization operator within the model 
# construction; along the way we will also prune the nnll list.
@time 𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = nnll_spec, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real, )

@show length(𝔹basis)            

##

@profview let mb_spec = nnll_spec, Rnl_spec = Rn_spec, Ylm_spec = Ylm_spec, basis = real
   𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = mb_spec, 
            Rnl_spec = Rnl_spec, Ylm_spec = Ylm_spec, basis = basis )
end
