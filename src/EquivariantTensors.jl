module EquivariantTensors

abstract type AbstractETLayer end 

using Bumper, WithAlloc, Random, GPUArraysCore

import ACEbase: evaluate, evaluate!
import WithAlloc: whatalloc
import ChainRulesCore: rrule, frule 

using ForwardDiff: Dual, extract_derivative 

export O3 

include("generics.jl")


# ------------------------------------------------------
# Core ACE model functionality 
include("ace/static_prod.jl")
include("ace/sparseprodpool.jl")
include("ace/sparseprodpool_ka.jl")
include("ace/sparsesymmprod.jl")
include("ace/sparsesymmprod_ka.jl")
include("ace/sparse_ace.jl")
include("ace/sparse_ace_utils.jl")


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

end
