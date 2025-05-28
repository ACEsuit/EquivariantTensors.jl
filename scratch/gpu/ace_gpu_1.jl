# First prototype implementation of a pure GPU ACE implementation. 
#

using GPUArrays, SparseArrays, LinearAlgebra, StaticArrays, Metal, LuxCore, 
      Combinatorics, Random, KernelAbstractions

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

dev = MtlArray

module ACEKA 

   using GPUArrays, SparseArrays, LinearAlgebra, StaticArrays, Metal, LuxCore, 
         Combinatorics

   import EquivariantTensors as ET 
   import Polynomials4ML as P4ML      
   import KernelAbstractions as KA
   import LuxCore: initialparameters, initialstates

   struct SimpleACE{T, RB, YB, BB}
      rbasis::RB      # radial embedding Rn
      ybasis::YB      # angular embedding Ylm
      symbasis::BB    # symmetric basis 
      params::Vector{T}   # model parameters
   end

   
end

function rand_graph(nnodes;
                    nneigrg = 20:40,  
                    T = Float32, 
                    rcut = one(T))
   ii = Int[] 
   jj = Int[]
   R = SVector{3, T}[]
   rmax = nnodes^(1/3) * 0.5
   maxneigs = 0 
   for i in 1:nnodes
      nneig = rand(nneigrg)
      maxneigs = max(maxneigs, nneig)
      neigs_i = shuffle(1:nnodes)[1:nneig] 
      for t in 1:nneig
         push!(ii, i)
         push!(jj, neigs_i[t])
         u = randn(SVector{3, T})
         r = (0.001 + rand() * rcut) / (0.001 + rmax) 
         push!(R, r * u / norm(u))
      end
   end
   return ii, jj, R, maxneigs 
end

function reshape_embedding(P, ii, jj, nnodes, maxneigs)
   @kernel function _reshape_embedding!(P3, P, ii, jj, nnodes, maxneigs)
      a, ifeat = @index(Global, NTuple)
      i = ii[a]
      j = jj[a] 
      P3[j, i, ifeat] = P[a, ifeat]
      nothing 
   end
   
   nfeatures = size(P, 2)
   P3 = similar(P, (maxneigs, nnodes, nfeatures))
   fill!(P3, zero(eltype(P3)))
   backend = KernelAbstractions.get_backend(P3)
   kernel! = _reshape_embedding!(backend)
   kernel!(P3, P, ii, jj, nnodes, maxneigs; ndrange = size(P))
   return P3
end

##
# generate an input graph 

ii, jj, R, maxneigs = rand_graph(100)
ii_dev = dev(ii); jj_dev = dev(jj); R_dev = dev(R)
maxneigs

# transform the relative positions R into radial and angular components 
# this also incorporates a radial distance transformation to put the range of 
# r into an admissible interval ... 
r = map(𝐫 -> 1 / (1 + norm(𝐫)), R) 
r_dev = map(𝐫 -> 1 / (1 + norm(𝐫)), R_dev) 
R̂ = map(𝐫 -> 𝐫 / norm(𝐫), R)
R̂_dev = map(𝐫 -> 𝐫 / norm(𝐫), R_dev) 

## 
# generate the input layer 

# Some model parameters that we will use: 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)

##
# first specify the radial and angular embeddings 
rbasis = P4ML.ChebBasis(Dtot+1)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)

Rn_dev = P4ML.evaluate(rbasis, r_dev)
Ylm_dev = P4ML.evaluate(ybasis, R̂_dev)

# now we need to reshape this output into a format suitable for the 
ntot = length(ii) 
nnodes = maximum(ii)
Rn_dev_3 = reshape_embedding(Rn_dev, ii_dev, jj_dev, nnodes, maxneigs)
Ylm_dev_3 = reshape_embedding(Ylm, ii_dev, jj_dev, nnodes, maxneigs)

# Rn_dev_3, Ylm_dev_3 are now in a format that is nice for the abasis 
# hence the LinearACE basis layer. We quickly construct a toy model basis 

mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = mb_spec, 
            Rnl_spec = P4ML.natural_indices(rbasis), 
            Ylm_spec = P4ML.natural_indices(ybasis), 
            basis = real )

# evaluate the A basis             
abasis = 𝔹basis.abasis
aspec_dev = dev(abasis.spec)
A = ET.ka_evaluate(abasis, (Rn_dev_3, Ylm_dev_3), aspec_dev)

# evaluate the AA basis
aabasis = 𝔹basis.aabasis
aaspecs_dev = dev.(aabasis.specs)
AA = ET.ka_evaluate(aabasis, A, aaspecs_dev)

# evaluate the O(3)-invariant basis. Here we have a problem - there is no 
# generic fallback of sparse matrix multiplication available? 
𝒞 = Float32.(𝔹basis.A2Bmaps[1])
𝒞_dev = Metal.mtl(𝒞)
