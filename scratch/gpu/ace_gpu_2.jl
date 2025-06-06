# First prototype implementation of a pure GPU ACE implementation. 
#

using GPUArrays, SparseArrays, LinearAlgebra, StaticArrays, Metal, LuxCore, 
      Combinatorics, Random, KernelAbstractions, LuxCore, MLDataDevices

import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import KernelAbstractions as KA

dev = gpu_device() 

module ACEKA

   using GPUArrays, SparseArrays, LinearAlgebra, StaticArrays, Metal, LuxCore, 
         Combinatorics, KernelAbstractions, Random, 
         MLDataDevices 
   using MLDataDevices: AbstractDevice         

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

   initialparameters(rng::AbstractRNG, m::SimpleACE) = 
            ( symbasis = initialparameters(rng, m.symbasis), 
                params = copy(m.params), )

   initialstates(rng::AbstractRNG, m::SimpleACE) = 
            ( symbasis = initialstates(rng, m.symbasis), )

   initialparameters(rng::AbstractRNG, bas::ET.SparseACE) = 
            NamedTuple() 

   initialstates(rng::AbstractRNG, bas::ET.SparseACE) = 
            ( aspec = bas.abasis.spec, 
              aaspecs = bas.aabasis.specs, 
              A2Bmaps = ET.DevSparseMatrixCSR.(bas.A2Bmaps), )

   # --------------------------------- 
   #  input management               

   struct PtClGraph{VECI, VECR}
      ii::VECI   # source indices
      jj::VECI   # target indices
      R::VECR  # relative positions
      nnodes::Int       # number of nodes in the graph
      maxneigs::Int     # maximum number of neighbors per node
   end

   (dev::Type{<: AbstractGPUArray})(X::PtClGraph) = 
      PtClGraph(dev(X.ii), dev(X.jj), dev(X.R), X.nnodes, X.maxneigs)

   (dev::AbstractDevice)(X::PtClGraph) = 
      PtClGraph(dev(X.ii), dev(X.jj), dev(X.R), X.nnodes, X.maxneigs)

   # ---------------------------------
   # evaluation code    

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

   function evaluate(model::SimpleACE, X::PtClGraph, 
                     ps, st)
      # transform the relative positions R into radial and angular components 
      # this also incorporates a radial distance transformation to put the range of 
      # r into an admissible interval ... 
      r = map(𝐫 -> 1 / (1 + norm(𝐫)), X.R) 
      R̂ = map(𝐫 -> 𝐫 / norm(𝐫), X.R)
      Rn = P4ML.evaluate(model.rbasis, r)
      Ylm = P4ML.evaluate(model.ybasis, R̂)

      # now we need to reshape these embeddings into a format suitable to 
      # apply the pooling operation 
      Rn_3 = reshape_embedding(Rn, X.ii, X.jj, X.nnodes, X.maxneigs)
      Ylm_3 = reshape_embedding(Ylm, X.ii, X.jj, X.nnodes, X.maxneigs)

      # Rn_3, Ylm_3 are now in a format that is nice for the abasis 
      # hence the LinearACE basis layer. The following evaluation should be 
      # moved into the ET module, for now just a quick prototype: 
      𝔹, _ = ET.ka_evaluate(model.symbasis, Rn_3, Ylm_3, ps.symbasis, st.symbasis)

      # apply the parameters 
      return transpose(𝔹) * ps.params, st 
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
   graph = ACEKA.PtClGraph(ii, jj, R, nnodes, maxneigs)
end


##
# generate a model 
Dtot = 16   # total degree; specifies the trunction of embeddings and correlations
maxl = 10    # maximum degree of spherical harmonics 
ORD = 3     # correlation-order (body-order = ORD + 1)
rbasis = P4ML.ChebBasis(Dtot+1)
ybasis = P4ML.real_solidharmonics(maxl; T = Float32, static=true)
mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)
𝔹basis = ET.sparse_equivariant_tensor(; 
            L = 0, mb_spec = mb_spec, 
            Rnl_spec = P4ML.natural_indices(rbasis), 
            Ylm_spec = P4ML.natural_indices(ybasis), 
            basis = real )
θ = randn(Float32, length(𝔹basis, 0))

model = ACEKA.SimpleACE(rbasis, ybasis, 𝔹basis, θ)
ps, st = LuxCore.setup(MersenneTwister(1234), model)

##
# test evaluation 

# 1. generate a random input graph 
X = rand_graph(100)

# 2. Move model and input to the GPU / Device 
ps_dev = dev(ps)
st_dev = dev(st)
X_dev = dev(X)

# 3. run forwardpass through the model
φ, _ = ACEKA.evaluate(model, X_dev, ps_dev, st_dev) 

