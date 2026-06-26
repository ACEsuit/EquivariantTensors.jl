
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

Coupling operators are constructed using the efficient kernel method proposed in [1]. At present, the only implementation is for O(3), in the `O3` module.

### O(3) implementation

#### O(3)-related symmetries

Let

```math
R = (r_1, \ldots, r_N) \in \Omega^N.
```

Here ``\Omega`` denotes the one-particle domain.

A function

```math
F : \Omega^N \to \mathbb{F}^{2L+1}
```

is group-equivariant (GE) if

```math
F(Q \cdot R) = D^L(Q) F(R), \qquad Q \in SO(3).
```

Here ``D^L`` denotes the Wigner-D matrix of order `L`.

It is permutation-invariant (PI) if

```math
F(R_\sigma) = F(R), \qquad \sigma \in S_N.
```

Combining GE and PI gives the group-equivariant and permutation-invariant (GE-PI) condition

```math
F(Q \cdot R_\sigma) = D^L(Q) F(R).
```

Equivalently, PI functions can be viewed as functions on multisets ``MS(\Omega)``. In that setting, the number of elements in ``R`` is allowed to vary.

Reflection parity is described by inversion,

```math
F(-R) = (-1)^p F(R),
```

where

```math
p = 0 \quad \text{for reflection-symmetric/even functions}, \qquad
p = 1 \quad \text{for reflection-antisymmetric/odd functions}.
```

#### Equivariant basis construction

Denote the index tuples by

```math
\mathbf{n} = (n_1, \ldots, n_N), \qquad
\mathbf{l} = (l_1, \ldots, l_N), \qquad
\mathbf{m} = (m_1, \ldots, m_N).
```

A typical compatible one-particle basis function has the form

```math
\phi_{n_i l_i m_i}(r_i)
=
R_{n_i}(|r_i|)Y_{l_i m_i}(\hat r_i),
```

possibly with additional non-angular labels absorbed into the non-angular index. In the ordered GE case (`PI=false`), the tensor product basis is

```math
\mathbf{A}^{\mathrm{GE}}_{\mathbf{n}\mathbf{l}\mathbf{m}}(R)
=
\prod_{i=1}^N
\phi_{n_i l_i m_i}(r_i).
```

For the GE-PI case (`PI=true`), pooling gives the one-particle features

```math
A_{n l m}(R) = \sum_j \phi_{n l m}(r_j),
```

and the tensor product basis is

```math
\mathbf{A}^{\mathrm{GE-PI}}_{\mathbf{n}\mathbf{l}\mathbf{m}}(R)
=
\prod_{i=1}^N
A_{n_i l_i m_i}(R).
```

This matches the convention used by the building blocks above, where `A` and `AA` refer to the one-particle features and their tensor products, respectively.

The coupled basis functions with target order `L` are chosen as linear combinations

```math
B^{\mathbf{n},\mathbf{l},L}_{a}(R)
=
\sum_{\mathbf{m}}
C^{\mathbf{n},\mathbf{l},L}_{a,\mathbf{m}}
\mathbf{A}_{\mathbf{n}\mathbf{l}\mathbf{m}}(R).
```

Here the tensor product basis `\mathbf{A}` is chosen according to the value of `PI`.

The expansion coefficients `C` are chosen so that the resulting `B` basis satisfies GE or GE-PI, with the requested reflection parity. They are returned by [`O3.coupling_coeffs`](@ref EquivariantTensors.O3.coupling_coeffs).

The arguments are:

* `L`: target equivariance order.
* `ll`: tuple representing ``\mathbf{l}``.
* `nn`: tuple representing ``\mathbf{n}``; if provided, it must have the same length as `ll`.
* `PI`: whether the coupled basis should also satisfy PI; default is `true` when `nn` is provided and `false` otherwise.
* `basis`: spherical harmonics convention in the one-particle basis; default is `complex`, and `real` follows `SpheriCart.jl`.
* `refl_sym`: reflection parity; `:sym` selects even parity and `:asym` selects odd parity. The default `nothing` chooses the parity compatible with the output order `L`.

The function can be called as, e.g.,

```julia
C, MM = O3.coupling_coeffs(L, ll, nn; PI=true, basis=real)
```

The outputs are:

* `MM` is the list of magnetic-index tuples appearing in the expansion.
* `C` contains the coupling coefficients in the expansion formula above. Rows of `C` index the independent coupled basis functions (index `a` in the above formula for the `B` basis). Columns of `C` correspond to entries of `MM` (index ``\mathbf{m}``). For `L = 0`, entries of `C` are scalars, while for `L > 0`, they are stored as `SVector{2L+1}`.

### References

[1] Efficient construction and explicit dimensionality of Lie group-equivariant and permutation-invariant spaces, https://arxiv.org/pdf/2604.01975.


## Pre-built Equivariant Tensors

