
using LinearAlgebra, Lux, Random, Test, Zygote, StaticArrays, ForwardDiff
using ACEbase.Testing: print_tf, println_slim
using Optimisers: destructure

using EquivariantTensors
import EquivariantTensors as ET 
import Polynomials4ML as P4ML      
import ForwardDiff as FD 
import DecoratedParticles as DP 

# include(joinpath(@__DIR__(), "..", "test_utils", "utils_gpu.jl"))

include("luxtestmodels.jl")
include("diffutils.jl")

using Metal 
gpu = dev = mtl 
Metal.versioninfo()

##

# build some models 
Dtot = 8; maxl = 6; ORD = 3

embed, acel = LTM.build_model(; Dtot, maxl, ORD, LL = (0, ), NFEAT = (2,))

model_0_2 = Lux.Chain(; embed = embed, ace = acel, 
         readout = WrappedFunction( 
                  U -> sum(U[1][:,1]) + 0.1f0 * sum(U[1][:,2].^2) ) )

embed_1_1, acel_1_1 = LTM.build_model(; Dtot, maxl, ORD, LL = (1, ), NFEAT = (1,))

model_1_1 = Lux.Chain(; embed = embed_1_1, ace = acel_1_1, 
         readout = WrappedFunction( U -> sum(abs2, sum(U[1])) ) )


embed_01, acel_01 = LTM.build_model(; Dtot, maxl, ORD, LL = (0, 1), NFEAT = (3, 1))
model_01 = Lux.Chain(; embed = embed_01, ace = acel_01, 
         readout = WrappedFunction( U -> (U0=U[1]; U1 = U[2]; 
            sum(U0[:,1]) + 0.1f0 * sum(U0[:,2] .^ 2) + 0.01f0 * sum(U0[:,3].^3)
            + sum(abs2, sum(U1)) ) 
         ) # WrappedFunction 
      ) # Chain 

# generate a random input graph 
nnodes = 30
X = ET.Testing.rand_graph(nnodes; nneigrg = 5:10)         

##

# model = model_0_2    # invariant, 2 features
# model = model_1_1    # equivariant, single feature
model = model_01   # 3 x invariant + 1 x equivariant feature

ps, st = LuxCore.setup(MersenneTwister(1234), model)
ps = ET.float32(ps); st = ET.float32(st)

## 

@info("test model evaluation on CPU")
φ, _ = model(X, ps, st)

@info("Test differentiation via Zygote")
g_zy = DIFF.grad_zy(X, model, ps, st)

@info("Test differentiation via ForwardDiff")
g_fd = DIFF.grad_fd(X, model, ps, st)

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
g_zy_dev = DIFF.grad_zy(X_dev, model, ps_dev, st_dev)
g_zy_32 = DIFF.grad_zy(ET.float32(X), model, ET.float32(ps), ET.float32(st))

g_zy_dev_e = Array(g_zy_dev.edge_data)
g_zy_32_e = Array(g_zy_32.edge_data)
@info("confirm matching gradients on CPU and GPU")
@error("This test currently fails for some models!") 
@show all(g_zy_dev_e .≈ g_zy_32_e)
_errs = norm.(g_zy_dev_e - g_zy_32_e)
@show sum(_errs) / length(_errs)
@show norm(_errs) / sqrt(length(_errs))
@show maximum(_errs)

## 
# Check gradient w.r.t. parameters 

@info("Test gradient w.r.t. parameters")

g_ps = DIFF.grad_zy_ps(X, model, ps, st)
g_ps_32 = DIFF.grad_zy_ps(ET.float32(X), model, ET.float32(ps), ET.float32(st))
g_ps_dev = DIFF.grad_zy_ps(X_dev, model, ps_dev, st_dev)
g_WLL = g_ps.ace.WLL 
g_WLL_32 = g_ps_32.ace.WLL 
g_WLL_dev = g_ps_dev.ace.WLL

g_ps_fd = DIFF.grad_fd_ps(X, model, ps, st)
g_WLL_fd = g_ps_fd.ace.WLL

@info("confirm matching parameter gradients on CPU and GPU")
println_slim(@test all(g_WLL .≈ g_WLL_fd))
println_slim(@test all(Float32.(g_WLL[i]) ≈ g_WLL_32[i] for i = 1:length(g_WLL) ))

Array(g_WLL_dev[1]) ≈ g_WLL_32[1]

## 

@show typeof(g_WLL_dev)