#
# Correctness test: compare scatter/gather prodpool vs fused KA version
#
# Run:  julia --project=. tmp/test_scatter_prodpool.jl
#

using EquivariantTensors, Test, LinearAlgebra
using EquivariantTensors: PooledSparseProduct, evaluate,
   _generate_input, ka_evaluate, ka_pullback
using ACEbase.Testing: fdtest, print_tf

include(joinpath(@__DIR__, "scatter_prodpool.jl"))

# -------------------------------------------------------

function _generate_basis(; order = 3, len = 50)
   NN = [rand(10:30) for _ = 1:order]
   spec = sort([ntuple(t -> rand(1:NN[t]), order)
                for _ = 1:len])
   return PooledSparseProduct(spec)
end

function _make_batched(basis; nneig = rand(8:16),
                              nnodes = rand(4:8))
   order = length(basis.spec[1])
   _bBB = _generate_input(basis; nX = nneig * nnodes)
   BB = ntuple(
      i -> reshape(_bBB[i], (nneig, nnodes, :)), order)
   return BB, nneig, nnodes
end

# -------------------------------------------------------

@info("Test sg_evaluate vs ka_evaluate (batched)")
for ntest = 1:20
   order = mod1(ntest, 4)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)

   A_ka = ka_evaluate(basis, BB, basis.spec,
                      nnodes, nneig)
   A_sg = sg_evaluate(basis, BB;
                      nneig, nnodes)

   print_tf(@test A_ka ≈ A_sg)
end
println()

# -------------------------------------------------------

@info("Test sg_pullback vs ka_pullback (batched)")
for ntest = 1:20
   order = mod1(ntest, 4)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)

   A = sg_evaluate(basis, BB; nneig, nnodes)
   ∂A = randn(Float64, size(A))

   ∂BB_ka = ka_pullback(∂A, basis, BB, basis.spec,
                        nnodes, nneig)
   ∂BB_sg = sg_pullback(∂A, basis, BB;
                        nneig, nnodes)

   ok = all(∂BB_ka[t] ≈ ∂BB_sg[t]
            for t = 1:order)
   print_tf(@test ok)
end
println()

# -------------------------------------------------------

@info("Test sg_pullback via finite differences")
for ntest = 1:20
   order = mod1(ntest, 4)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)
   gidx = GatherIndices(basis, BB)

   A = sg_evaluate(basis, BB; gidx, nneig, nnodes)
   ∂A = randn(Float64, size(A))

   UU = ntuple(
      i -> randn(Float64, size(BB[i])), order)
   _BB(t) = ntuple(
      i -> BB[i] + t * UU[i], order)

   F(t) = dot(∂A,
      sg_evaluate(basis, _BB(t);
                  nneig, nnodes))
   dF(t) = begin
      ∂BB = sg_pullback(∂A, basis, _BB(t);
                        nneig, nnodes)
      sum(dot(∂BB[i], UU[i]) for i = 1:order)
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

# -------------------------------------------------------

@info("Test GatherIndices precomputation")
for ntest = 1:10
   order = mod1(ntest, 3)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)
   gidx = GatherIndices(basis, BB)

   # Check that calling with gidx gives same result
   A1 = sg_evaluate(basis, BB; nneig, nnodes)
   A2 = sg_evaluate(basis, BB;
                    gidx, nneig, nnodes)
   print_tf(@test A1 ≈ A2)
end
println()

# -------------------------------------------------------

@info("Test fs_pullback vs ka_pullback (batched)")
for ntest = 1:20
   order = mod1(ntest, 4)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)

   A = sg_evaluate(basis, BB; nneig, nnodes)
   ∂A = randn(Float64, size(A))

   ∂BB_ka = ka_pullback(∂A, basis, BB, basis.spec,
                        nnodes, nneig)
   ∂BB_fs = fs_pullback(∂A, basis, BB;
                        nneig, nnodes)

   ok = all(∂BB_ka[t] ≈ ∂BB_fs[t]
            for t = 1:order)
   print_tf(@test ok)
end
println()

# -------------------------------------------------------

@info("Test fs_pullback via finite differences")
for ntest = 1:20
   order = mod1(ntest, 4)
   basis = _generate_basis(; order)
   BB, nneig, nnodes = _make_batched(basis)

   A = sg_evaluate(basis, BB; nneig, nnodes)
   ∂A = randn(Float64, size(A))

   UU = ntuple(
      i -> randn(Float64, size(BB[i])), order)
   _BB(t) = ntuple(
      i -> BB[i] + t * UU[i], order)

   F(t) = dot(∂A,
      sg_evaluate(basis, _BB(t);
                  nneig, nnodes))
   dF(t) = begin
      ∂BB = fs_pullback(∂A, basis, _BB(t);
                        nneig, nnodes)
      sum(dot(∂BB[i], UU[i]) for i = 1:order)
   end
   print_tf(@test fdtest(F, dF, 0.0; verbose = false))
end
println()

println("\nAll tests passed.")
