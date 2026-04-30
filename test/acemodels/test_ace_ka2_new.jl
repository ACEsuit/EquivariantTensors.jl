
using LinearAlgebra, Lux, Random, Test, Zygote, StaticArrays, ForwardDiff
using ACEbase.Testing: print_tf, println_slim
using Optimisers: destructure

using EquivariantTensors
import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import ForwardDiff as FD 
import DecoratedParticles as DP 

# include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

using Metal 
global gpu = global dev = mtl 
Metal.versioninfo()

##

Dtot = 8
maxl = 6
ORD = 3 

# ------ particle embeddings ------- 
# radial edge embedding 
rbasis = P4ML.legendre_basis(Dtot+1)
Rn_spec = P4ML.natural_indices(rbasis) 
Rembed = ET.EdgeEmbed( ET.EmbedDP( 
                  ET.dp_transform( x -> 1 / (1 + norm(x.𝐫)) ), 
                  rbasis ) )

# angular edge embedding                  
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)
Yembed = ET.EdgeEmbed( 
               ET.EmbedDP( ET.dp_transform( x -> x.𝐫 ), 
                      ybasis ) )

# combine the radial and angular embeddings 
embed = Parallel(nothing; Rnl = Rembed, Ylm = Yembed)

# ------ ace basis specification ------- 
# generate the nnll basis pre-specification
nnll_long = ET.sparse_nnll_set(; ORD = ORD, 
                  minn = 0, maxn = Dtot, maxl = maxl, 
                  level = bb -> sum((b.n + b.l) for b in bb; init=0), 
                  maxlevel = Dtot)

# conver this into an ACE basis with both L = 0 and L = 1 features 
𝔹basis = ET.sparse_equivariant_tensors(; 
            LL = (0, 1), mb_spec = nnll_long, 
            Rnl_spec = Rn_spec, 
            Ylm_spec = Ylm_spec, 
            basis = real )

# the ace layer takes a basis and contracts it with learnable weights 
# to produce final outputs or output features: here we produce 
#     3 scalar features and 1 vector feature per node.
acel = ET.SparseACElayer(𝔹basis, (3, 1))

# to build a model, we combine the embedding and ace layers into a Lux Chain 
# to create a model we add a final layer that maps all the nodal output features 
# to a single scalar. 
readout = WrappedFunction( U01 -> ( U0 = U01[1]; U1 = U01[2]; 
                                    sum(U0[:,1]) + 0.1f0 * sum(U0[:,2].^2) 
                                         + 0.01f0 * sum(U0[:,3].^3) 
                                         + 0.234f0 * sum(x -> sum(abs2, x), U1) 
                                       ) )
model = Lux.Chain(; embed = embed, ace = acel, readout = readout )
ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)
θ_0 = ps.ace.WLL[1] # for testing only 

##
# test evaluation 

# 1. generate a random input graph 
nnodes = 30
X = ET.Testing.rand_graph(nnodes; nneigrg = 5:10)

# for a larger test 
# nnodes = 100
# X = ET.Testing.rand_graph(nnodes; nneigrg = 10:20)

@info("Basic ETGraph tests")
println_slim(@test ET.nnodes(X) == nnodes)
# println_slim(@test ET.maxneigs(X) <= 20)
println_slim(@test ET.nedges(X) == length(X.ii) == length(X.jj) == X.first[end] - 1)
println_slim(@test all( all(X.ii[X.first[i]:X.first[i+1]-1] .== i)
                        for i in 1:nnodes ) )

## 

@info("test model evaluation on CPU")
φ, _ = model(X, ps, st)

_grad_zy(X, model, ps, st) = Zygote.gradient(G -> model(G, ps, st)[1], X)[1]

@info("Test differentiation via Zygote")
g_zy = _grad_zy(X, model, ps, st)


function _grad_fd(G, model, ps, st) 

   function replace_edges(X, Rmat)
      Rsvec = [ SVector{3}(Rmat[:, i]) for i in 1:size(Rmat, 2) ]
      new_edgedata = [ (; 𝐫 = 𝐫) for 𝐫 in Rsvec ]
      return ET.ETGraph( X.ii, X.jj, X.first, 
                  X.node_data, new_edgedata, X.graph_data, 
                  X.maxneigs )
   end 

   function _eval_mat(Rmat)
      G_new = replace_edges(G, Rmat)
      return model(G_new, ps, st)[1]
   end
      
   Rsvec = [ x.𝐫 for x in G.edge_data ]
   Rmat = reinterpret(reshape, eltype(Rsvec[1]), Rsvec)
   ∇E_fd = ForwardDiff.gradient(_eval_mat, Rmat)
   ∇E_svec = [ SVector{3}(∇E_fd[:, i]) for i in 1:size(∇E_fd, 2) ]
   ∇E_edges = [ DP.VState(; 𝐫 = 𝐫) for 𝐫 in ∇E_svec ]
   return ET.ETGraph( G.ii, G.jj, G.first, 
               G.node_data, ∇E_edges, G.graph_data, 
               G.maxneigs )
end 

function _grad_fd_ps(G, model, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = model(G, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

@info("Test differentiation via ForwardDiff")
g_fd = _grad_fd(X, model, ps, st)

@info("test agreement of Zygote and ForwardDiff gradients")
println_slim(@test all(g_zy.edge_data .≈ g_fd.edge_data)) 

##


@info("Test model evaluation on GPU") 

# 2. Move model and input to the GPU / Device 
ps_dev = dev(ET.float32(ps))
st_dev = dev(ET.float32(st))
X_dev = dev(ET.float32(X))

# evaluate on CPU and GPU 
φ_dev, _ = model(X_dev, ps_dev, st_dev) 

@info("confirm matching forwardpass outputs on CPU and GPU")
println_slim(@test Float32(φ_dev) ≈ Float32(φ)) 

##
@info("evaluate X-gradient on GPU ")
g_zy_dev = _grad_zy(X_dev, model, ps_dev, st_dev)
g_zy_32 = _grad_zy(ET.float32(X), model, ET.float32(ps), ET.float32(st))

g_zy_dev_e = Array(g_zy_dev.edge_data)
g_zy_32_e = Array(g_zy_32.edge_data)
@info("confirm matching gradients on CPU and GPU")
@error("This test currently fails!") 
@show all(g_zy_dev_e .≈ g_zy_32_e)
_errs = norm.(g_zy_dev_e - g_zy_32_e)
@show sum(_errs) / length(_errs)
@show norm(_errs) / sqrt(length(_errs))
@show maximum(_errs)

## 
# Check gradient w.r.t. parameters 

@info("Test gradient w.r.t. parameters")

_grad_zy_ps(model, X, ps, st) = Zygote.gradient(_ps -> model(X, _ps, st)[1], ps)[1]
g_ps = _grad_zy_ps(model, X, ps, st)
g_ps_32 = _grad_zy_ps(model, ET.float32(X), ET.float32(ps), ET.float32(st))
g_ps_dev = _grad_zy_ps(model, X_dev, ps_dev, st_dev)
g_WLL = g_ps.ace.WLL 
g_WLL_32 = g_ps_32.ace.WLL 
g_WLL_dev = g_ps_dev.ace.WLL

g_ps_fd = _grad_fd_ps(X, model, ps, st)
g_WLL_fd = g_ps_fd.ace.WLL

@info("confirm matching parameter gradients on CPU and GPU")
println_slim(@test all(g_WLL .≈ g_WLL_fd))
println_slim(@test all(Float32.(g_WLL[i]) ≈ g_WLL_32[i] for i = 1:length(g_WLL) ))

Array(g_WLL_dev[1]) ≈ g_WLL_32[1]
