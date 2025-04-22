
# Public API

`EquivariantTensors` provides several building blocks for (neural and) tensor networks that preserve various symmetries. These can be combined into various tensor formats from which equivariant parameterized models can be built. 

## Building Blocks

* Fused tensor product and pooling [`PooledSparseProduct`](@ref)
* Sparse symmetric product [`SparseSymmProd`](@ref)

Evaluating a layer can be done both in-place or allocating, e.g., via 
```julia
abasis::SparseSymmProd
evaluate!(AA, abasis, A)
AA = evaluate(abasis, A)
```
We refer to the individual documentation for the details of the arguments to each layer. 

All tensor layers have custom pullbacks implemented that can be accessed via non-allocating or allocating calls, e.g., 
```julia 
pullback!(∂AA, ∂A, abasis, A)
∂AA = pullback(∂A, abasis, A)
```

Pushforwards and reverse-over-reverse are implemented using ForwardDiff. This is quasi-optimal even for reverse-over-reverse due to the fact that it can be interpreted as a directional derivative on evaluate and pullback (after swapping derivatives). As a matter of fact, we generally recommend to not use these directly. ChainRules integration would give an easier use-pattern. For optimal performance the same technique should be applied to an entire model architecture rather than to each individual layer. This would avoid several unnecessary intermediate allocations.

The syntax for pushforwards is straightforward:
```julia
pushforward!(P, ∂P, layer, X, ∂X)
P, ∂P = pushforward(layer, X, ∂X)
```

For second-order pullbacks the syntax is 
```julia
pullback2!(∇_∂P, ∇_X, ∂∂X, ∂P, layer, X)
∇_∂P, ∇_X = pullback2(∂∂X, ∂P, layer, X)
```

### Bumper and WithAlloc usage

Using the `WithAlloc.jl` interface the api can be used conveniently as follows (always from within a `@no_escape` block)
```julia 
A = @withalloc evaluate!(abasis, BB)
∂X = @withalloc pullback!(∂P, layer, X)
P, ∂P = @withalloc pushforward!(layer, X, ∂X)
```


## Lie Group Symmetrization / Coupling Operators

* Construct coupling coefficients: [`O3.coupling_coeffs`](@ref)

(TODO: detailed description)


## Pre-built Equivariant Tensors

