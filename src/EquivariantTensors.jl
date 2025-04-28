module EquivariantTensors

abstract type AbstractETLayer end 

using Bumper, WithAlloc, Random 

import ACEbase: evaluate, evaluate!
import WithAlloc: whatalloc
import ChainRulesCore: rrule, frule 

using ForwardDiff: Dual, extract_derivative 

export O3 


include("generics.jl")


include("ace/static_prod.jl")
include("ace/sparseprodpool.jl")
include("ace/sparsesymmprod.jl")
include("ace/sparse_ace.jl")
include("ace/sparse_ace_utils.jl")

include("yyvector.jl")
include("O3.jl")

include("utils/setproduct.jl")
include("utils/invmap.jl")
include("utils/sparse.jl")
include("utils/symmop.jl")

end
