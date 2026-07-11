using EquivariantTensors, StaticArrays, Test, LinearAlgebra
import EquivariantTensors as ET
import EquivariantTensors: O3
import SpheriCart
import Polynomials4ML as P4ML
using LuxCore: setup
using Random

include(joinpath(@__DIR__, "..", "..", "test_utils", "utils_testO3.jl"))
# provides: rYlm, eval_cheb, rand_ball, lm2idx (via SpheriCart), ...

## ---------------------------------------------------------------
#  helpers

# real, sphericart-ordered Ylm spec covering l = 0:maxl
ylm_spec(maxl) = P4ML.natural_indices(SpheriCart.SphericalHarmonics(maxl))

# radial spec that carries l  (n = 1:nmax for each l = 0:maxl)
rnl_spec_withl(nmax, maxl) = [ (n = n, l = l) for l = 0:maxl for n = 1:nmax ]
# radial spec without l
rnl_spec_nol(nR) = [ (n = n,) for n = 1:nR ]

# Build (Rnl, Ylm) 3-tensors of shape (nneig, nnode, feat) from a per-node
# list of neighbour vectors `Rs[i]`. The radial features are made rotation
# INVARIANT (they depend only on |𝐫|), so the whole map is O3-equivariant.
function build_embeddings(Rs, nR, maxl)
   nnode = length(Rs)
   nneig = maximum(length, Rs)
   Rnl = zeros(nneig, nnode, nR)
   Ylm = zeros(nneig, nnode, (maxl + 1)^2)
   for i = 1:nnode, (j, 𝐫) in enumerate(Rs[i])
      Rnl[j, i, :] .= eval_cheb(𝐫, nR)      # invariant (|𝐫| only)
      Ylm[j, i, :] .= rYlm(maxl, 𝐫)         # real sphericart SH
   end
   return Rnl, Ylm
end

## ---------------------------------------------------------------

@testset "construction & spec validation" begin
   maxl = 3
   maxn = 5
   Yspec = ylm_spec(maxl)

   # radial spec carrying l: only radials with l == L feed the L-output
   op = ET.EquivariantTensorProduct((0, 1, 2), rnl_spec_withl(maxn, maxl), Yspec)
   @test op isa ET.EquivariantTensorProduct{3}
   @test op.LL == (0, 1, 2)
   for (iL, L) in enumerate(op.LL)
      @test op.ranges[iL] == findall(r -> r.l == L, rnl_spec_withl(maxn, maxl))
   end

   # radial spec without l: every radial couples to every L
   opn = ET.EquivariantTensorProduct((0, 1), rnl_spec_nol(maxn), Yspec)
   @test all(opn.ranges[i] == collect(1:maxn) for i = 1:2)

   # Ylm_spec not in sphericart order -> rejected
   badY = [ (l = 0, m = 0), (l = 1, m = 0), (l = 1, m = -1), (l = 1, m = 1) ]
   @test_throws ErrorException ET.EquivariantTensorProduct(
                                    (0, 1), rnl_spec_nol(maxn), badY)

   # Ylm_spec too short to cover Lmax -> rejected
   @test_throws ErrorException ET.EquivariantTensorProduct(
                                    (0, 1, 2), rnl_spec_nol(maxn), ylm_spec(1))
end


@testset "evaluation vs reference" begin
   maxl, nmax = 3, 4
   LL = (0, 1, 2)
   op = ET.EquivariantTensorProduct(LL, rnl_spec_withl(nmax, maxl),
                                    ylm_spec(maxl))
   ps, st = setup(MersenneTwister(11), op)

   nneig, nnode, nR = 4, 3, nmax * (maxl + 1)
   Rnl = randn(nneig, nnode, nR)
   Ylm = randn(nneig, nnode, (maxl + 1)^2)
   𝔹 = ET.evaluate(op, Rnl, Ylm, ps, st)

   for (iL, L) in enumerate(LL)
      # this is the correct sphericart convention, but fragile. If sphericart 
      # indexing ever changes, this will fail. But it is also concisten with the 
      # implementation of the layer, so it will catch any changes in sphericart indexing.
      blk = lm2idx(L, -L):lm2idx(L, L)          # sphericart L-block
      BL = 𝔹[iL]
      @test eltype(BL) == SVector{2L + 1, Float64}
      @test size(BL) == (nneig, nnode, length(op.ranges[iL]))
      ok = true
      for j = 1:nneig, i = 1:nnode, (k, iR) in enumerate(op.ranges[iL])
         ref = SVector{2L + 1}(Rnl[j, i, iR] .* Ylm[j, i, blk])
         ok &= BL[j, i, k] ≈ ref
      end
      @test ok
   end
end


@testset "Lux interface" begin
   maxl = 2
   LL = (0, 1, 2)
   op = ET.EquivariantTensorProduct(LL, rnl_spec_nol(4), ylm_spec(maxl))
   ps, st = setup(MersenneTwister(22), op)
   @test ps == NamedTuple()
   @test haskey(st, :ranges) && haskey(st, :LL)

   nneig, nnode = 3, 2
   Rnl = randn(nneig, nnode, 4)
   Ylm = randn(nneig, nnode, (maxl + 1)^2)

   # callable returns (𝔹, st) and matches evaluate
   𝔹, st2 = op((Rnl, Ylm), ps, st)
   @test st2 === st
   𝔹ref = ET.evaluate(op, Rnl, Ylm, ps, st)
   @test all(𝔹[i] == 𝔹ref[i] for i = 1:length(𝔹))

   # a NamedTuple (Rnl = ..., Ylm = ...) input works too (splat order)
   𝔹nt, _ = op((Rnl = Rnl, Ylm = Ylm), ps, st)
   @test all(𝔹nt[i] == 𝔹ref[i] for i = 1:length(𝔹))
end


@testset "O3 equivariance" begin
   Random.seed!(1234)
   maxl = 3
   LL = (0, 1, 2)
   nnode, nneig = 2, 4

   for (label, Rspec, nR) in (
            ("radial with l", rnl_spec_withl(2, maxl), 2 * (maxl + 1)),
            ("radial without l", rnl_spec_nol(5), 5) )
      op = ET.EquivariantTensorProduct(LL, Rspec, ylm_spec(maxl))
      ps, st = setup(MersenneTwister(7), op)

      Rs = [ [ rand_ball() for _ = 1:nneig ] for _ = 1:nnode ]
      Rnl, Ylm = build_embeddings(Rs, nR, maxl)
      𝔹 = ET.evaluate(op, Rnl, Ylm, ps, st)

      # rotate the configuration and re-embed
      θ = π * rand(3)
      Q = O3.Q_from_angles(θ)
      Rs_rot = [ [ Q * 𝐫 for 𝐫 in nbr ] for nbr in Rs ]
      Rnl2, Ylm2 = build_embeddings(Rs_rot, nR, maxl)
      @test Rnl2 ≈ Rnl        # radial embedding is rotation-invariant

      𝔹2 = ET.evaluate(op, Rnl2, Ylm2, ps, st)

      # each L-output must transform by the real Wigner-D matrix D_L(Q)
      for (iL, L) in enumerate(LL)
         D = O3.D_from_angles(L, θ, real)
         ok = all( 𝔹2[iL][j, i, k] ≈ D * 𝔹[iL][j, i, k]
                   for j = 1:nneig, i = 1:nnode, k = 1:size(𝔹[iL], 3) )
         @test ok
      end
   end
end
