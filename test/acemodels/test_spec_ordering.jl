# `sparse_equivariant_tensor` must not depend on the order of `mb_spec`.
#
# `SparseSymmProd` re-sorts its spec by correlation order (sparsesymmprod.jl:39),
# while the columns of the symmetrisation matrix index 𝔸spec in the order it was
# given. If the two are not reconciled the basis comes out SILENTLY
# non-equivariant: right shapes, wrong values, nothing raised. Callers that build
# `mb_spec` order-grouped are unaffected, which is why this went unnoticed.
#
# Shuffling `mb_spec` must change nothing at all: same 𝔸spec, same coupling
# matrix, same basis size.

using LinearAlgebra, Random, Test
using ACEbase.Testing: print_tf, println_slim
import EquivariantTensors as ET
import Polynomials4ML as P4ML

##

@info("sparse_equivariant_tensor is invariant to mb_spec ordering")

Dtot = 8; maxl = 4; ORD = 3

mb_spec = ET.sparse_nnll_set(; L = 0, ORD = ORD, minn = 0, maxn = Dtot, maxl = maxl,
               level = bb -> sum((b.n + b.l) for b in bb; init=0), maxlevel = Dtot)

Rnl_spec = sort(unique([ (n = b.n, l = b.l) for bb in mb_spec for b in bb ]))
Ylm_spec = P4ML.natural_indices(P4ML.real_solidharmonics(maxl))

build(spec) = ET.sparse_equivariant_tensor(; L = 0, mb_spec = spec,
                  Rnl_spec = Rnl_spec, Ylm_spec = Ylm_spec, basis = real)

sorted   = sort(mb_spec, by = length)
shuffled = shuffle(MersenneTwister(3), sorted)
# the test would be vacuous if the shuffle happened to stay order-grouped
println_slim(@test !issorted(shuffled, by = length))

Bs, Bsh = build(sorted), build(shuffled)

# the returned 𝔸spec must be order-grouped whatever was passed in -- this is the
# invariant SparseSymmProd silently assumes
println_slim(@test issorted(Bs.meta["𝔸spec"], by = length))
println_slim(@test issorted(Bsh.meta["𝔸spec"], by = length))

# and the two constructions must agree exactly
println_slim(@test Bs.meta["𝔸spec"] == Bsh.meta["𝔸spec"])
println_slim(@test Bs.meta["Aspec"] == Bsh.meta["Aspec"])
println_slim(@test size(Bs.A2Bmaps[1]) == size(Bsh.A2Bmaps[1]))

# Rows of A2B are the 𝔹 basis functions and follow `mb_spec` -- callers index
# their coefficients by that order, so it must NOT be canonicalised. Columns
# index 𝔸 and must be. So the two maps agree up to a permutation of rows:
rowset(A) = sort([ Vector(A[i, :]) for i = 1:size(A, 1) ])
println_slim(@test rowset(Bs.A2Bmaps[1]) == rowset(Bsh.A2Bmaps[1]))
