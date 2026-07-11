
# ET weight-initializer leaf samplers. `Random`-only — no `Lux` /
# `WeightInitializers` dependency. They follow the `WeightInitializers` calling
# convention (`init(rng, [T], dims...)`) so they drop straight into LuxCore/Lux
# `init_*` kwargs and `initialparameters`.
#
# They exist as a *separate ET surface* (rather than reusing `WeightInitializers`)
# on purpose: ET tensor formats are *multiplicative / contractive* in their
# parameters (CP = product of N factors, TT = a chain contraction, Tucker =
# factor·core), so the right initialisation scaling is generally NOT an MLP-style
# additive `1/fan` rule. These functions are only the *dumb leaf samplers*; the
# format-aware policy that maps a target output/coefficient scale to a
# per-parameter `σ` via each format's contraction structure is future work —
# see `agents/initializers.md`.

"""
   et_zeros([rng], [T=Float64], dims...) -> Array{T}

Zero initializer. An `rng` is accepted (and ignored) so it shares the calling
convention of the random initializers.
"""
et_zeros(::AbstractRNG, ::Type{T}, dims::Integer...) where {T} = zeros(T, dims...)
et_zeros(rng::AbstractRNG, dims::Integer...) = et_zeros(rng, Float64, dims...)

"""
   et_normal([rng], [T=Float64], dims...; σ = 1) -> Array{T}

Normal initializer: i.i.d. `N(0, σ²)` entries of element type `T`. The default
`σ = 1` is the *raw* sampler scale; a caller that knows its contraction
structure should pass an appropriate `σ` (e.g. `1/sqrt(fan_in)` for a linear
readout). See the module note above on why ET does not default to an MLP-style
scaling rule.
"""
et_normal(rng::AbstractRNG, ::Type{T}, dims::Integer...; σ = 1) where {T} =
      T(σ) .* randn(rng, T, dims...)
et_normal(rng::AbstractRNG, dims::Integer...; σ = 1) =
      et_normal(rng, Float64, dims...; σ = σ)
