module EquivariantTensors

abstract type AbstractETLayer end 

using Bumper, WithAlloc, Random, GPUArraysCore, KernelAbstractions

import ACEbase: evaluate, evaluate!, evaluate_ed, evaluate_ed!, 
                pullback, pullback!, pushforward, pushforward!
import WithAlloc: whatalloc
import ChainRulesCore: rrule, frule 
import LuxCore: initialparameters, initialstates, AbstractLuxLayer
import MLDataDevices: gpu_device, cpu_device 

import Polynomials4ML as P4ML

using ForwardDiff: Dual, extract_derivative

export O3, gpu_device, cpu_device  


include("generics.jl")

# ------------------------------------------------------
# embedding layers, transforms, & auxiliary functionality 
include("transforms/decpart.jl")
include("transforms/agnesi.jl")

include("embed/graph.jl")
include("embed/embeddings.jl")
include("embed/transsplines.jl")

# ------------------------------------------------------
# static product kernels, shared by pooling and the sparse format
include("utils/static_prod.jl")

# ------------------------------------------------------
# pooling layer: per-edge embeddings -> A
include("pooling/sparseprodpool.jl")
include("pooling/sparseprodpool_ka.jl")

# ------------------------------------------------------
# sparse tensor format: symmetric products + symmetrisation
# (symmprod_dag*.jl live here too but are currently not included)
include("formats/sparse/sparsesymmprod.jl")
include("formats/sparse/sparsesymmprod_ka.jl")
include("formats/sparse/sparse_ace_basis.jl")
include("formats/sparse/sparse_ace_layer.jl")
include("formats/sparse/sparse_ace_ka.jl")
include("formats/sparse/sparse_ace_utils.jl")
include("formats/sparse/sparsemat_ka.jl")

# ------------------------------------------------------
# groups: O3 irreps, CG coupling, carrier symmetrisation
include("groups/O3/O3.jl")
# (O3.jl includes O3_utils.jl, yyvector.jl, O3_transformations.jl,
#  quad_O3_data.jl, quad_O3.jl)
include("groups/symmop.jl")


# ------------------------------------------------------
# model building utilities 
include("utils/setproduct.jl")
include("utils/invmap.jl")
include("utils/sparseprod.jl")
include("utils/promotion.jl")

# a linear layer that selects a linear operator from 
# multiple choices depending on the input. 
include("utils/selectlinl.jl")
include("utils/selector.jl")

# other utilities 
#  adapt.jl : provides some conversion utilities especially moving 
#             Float64 to Float32 recursively in NamedTuples etc.
include("utils/adapt.jl")

# ------------------------------------------------------
# extensions 
include("extensions/atoms.jl")

# ------------------------------------------------------
# Testing utilities 
include("testing/testing.jl")


end
