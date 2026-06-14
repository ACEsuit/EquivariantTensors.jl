
# Unit tests for the weight-initializer leaf samplers (src/utils/initializers.jl)

using EquivariantTensors, Random, Test
import EquivariantTensors as ET

@info("Weight-initializer leaf samplers (et_zeros / et_normal)")

rng = MersenneTwister(0)

_rms(A) = sqrt(sum(abs2, A) / length(A))   # ≈ σ for zero-mean samples

## et_zeros
@test ET.et_zeros(rng, 3, 4) == zeros(3, 4)
@test eltype(ET.et_zeros(rng, 3, 4)) == Float64            # default eltype
@test eltype(ET.et_zeros(rng, Float32, 2)) == Float32      # explicit eltype

## et_normal: shape and eltype policy (default Float64, opt-in Float32)
@test size(ET.et_normal(rng, 5, 2)) == (5, 2)
@test eltype(ET.et_normal(rng, 5, 2)) == Float64
@test eltype(ET.et_normal(rng, Float32, 5)) == Float32

## σ scaling: sample RMS ≈ σ
@test isapprox(_rms(ET.et_normal(MersenneTwister(1), 4000; σ = 0.3)), 0.3; rtol = 0.1)
@test isapprox(_rms(ET.et_normal(MersenneTwister(2), 4000)), 1.0; rtol = 0.1)   # default σ = 1

## rng determinism
@test ET.et_normal(MersenneTwister(7), 3, 3) == ET.et_normal(MersenneTwister(7), 3, 3)
