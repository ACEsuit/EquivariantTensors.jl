module EquivariantTensors

abstract type AbstractETLayer end 

using Bumper, WithAlloc, Random, GPUArraysCore, KernelAbstractions

import ACEbase: evaluate, evaluate!, evaluate_ed, evaluate_ed!, 
                pullback, pullback!, pushforward, pushforward!
import WithAlloc: whatalloc
import ChainRulesCore: rrule, frule 
import LuxCore: initialparameters, initialstates, AbstractLuxLayer
import MLDataDevices: gpu_device, cpu_device 

using ForwardDiff: Dual, extract_derivative 

export O3, gpu_device, cpu_device  


include("generics.jl")

# ------------------------------------------------------
# embedding layers & auxiliary functionality 
include("embed/diffnt.jl")
include("embed/transform.jl")
include("embed/graph.jl")
include("embed/rnlylmbasis.jl")
include("embed/embeddings.jl")
include("embed/envelope.jl")


# ------------------------------------------------------
# Core ACE model functionality 
include("ace/static_prod.jl")
include("ace/sparseprodpool.jl")
include("ace/sparseprodpool_ka.jl")
include("ace/sparsesymmprod.jl")
include("ace/sparsesymmprod_ka.jl")
include("ace/sparse_ace_basis.jl")
include("ace/sparse_ace_layer.jl")
include("ace/sparse_ace_ka.jl")
include("ace/sparse_ace_utils.jl")
include("ace/sparsemat_ka.jl")

# ------------------------------------------------------
# O3 symmetrization
include("O3/O3.jl")
# O3/O3_transformations.jl
# O3/yyvector.jl 
# O3/O3_utils.jl 


# ------------------------------------------------------
# model building utilities 
include("utils/setproduct.jl")
include("utils/invmap.jl")
include("utils/sparseprod.jl")
include("utils/symmop.jl")
include("utils/promotion.jl")
include("utils/selectlinl.jl")

# other utilities 
include("utils/adapt.jl")

# ------------------------------------------------------
# extensions 
include("extensions/atoms.jl")

# ------------------------------------------------------
# Testing utilities 
include("testing/testing.jl")


end
