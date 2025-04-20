module EquivariantTensors

abstract type AbstractETLayer end 

using Bumper, WithAlloc, Random 

import ACEbase: evaluate, evaluate!
import WithAlloc: whatalloc
using ForwardDiff: Dual, extract_derivative 

function pullback end 
function pullback! end 
function pullback2 end 
function pullback2! end 
function pushforward end 
function pushforward! end 

export O3 

include("utils.jl")
include("ace/sparseprodpool.jl")


include("yyvector.jl")
include("O3.jl")

end
