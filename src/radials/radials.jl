module Radials

# Self-contained learnable / splined radial basis (`Rnl`) submodule, moved
# here from ACEpotentials `src/models/`. See `agents/radials.md` for the
# design and porting notes.

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

# element <-> index and SMatrix helpers live in ET's utils
import ..EquivariantTensors: _i2z, _z2i, _get_nz, _convert_zlist, _make_smatrix

export LearnableRnlBasis, SplineRnlBasis, splinify, learnable_Rnl_basis,
       PolyEnvelope1sR, PolyEnvelope2sX, agnesi_transform

include("envelopes.jl")
include("transforms.jl")
include("Rnl_learnable.jl")
include("Rnl_splines.jl")
include("splinify.jl")
include("constructors.jl")

end
