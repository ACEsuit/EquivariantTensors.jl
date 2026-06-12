module ACEradials

# Self-contained learnable / splined radial basis (`Rnl`) package, moved
# here from ACEpotentials `src/models/`. See `agents/radials.md` in the
# repository root for the design, porting notes, and the decision record
# for hosting this as a subdir package of EquivariantTensors.jl.

using StaticArrays: SMatrix, SVector
using Random: AbstractRNG
using Lux: glorot_normal

import ForwardDiff
using ForwardDiff: Dual

import ChainRulesCore: rrule, NoTangent, unthunk
import WithAlloc: whatalloc
import Polynomials4ML as P4ML

import ACEbase: evaluate, evaluate!, evaluate_ed
import LuxCore: AbstractLuxLayer, initialparameters, initialstates,
                parameterlength

export LearnableRnlBasis, SplineRnlBasis, splinify, learnable_Rnl_basis,
       PolyEnvelope1sR, PolyEnvelope2sX, agnesi_transform

include("elements.jl")
include("envelopes.jl")
include("transforms.jl")
include("agnesi_dp.jl")
include("Rnl_learnable.jl")
include("Rnl_splines.jl")
include("splinify.jl")
include("constructors.jl")

end
