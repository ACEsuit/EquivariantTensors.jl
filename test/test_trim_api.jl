using EquivariantTensors, Test
import EquivariantTensors as ET
include(joinpath(@__DIR__, "test_utils", "trim_cases.jl"))

@testset "Val(L) construction API == Integer API" begin
   for (maxn, maxl, ord) in TRIM_CASES
      mb = trim_mb_spec(maxn, maxl, ord); R = trim_rnl(maxn, maxl); Y = trim_ylm(maxl)
      S1, A1 = ET.symmetrisation_matrix(0, mb; prune = true, PI = true, basis = real)
      S2, A2 = ET.symmetrisation_matrix(Val(0), mb; prune = true, PI = true, basis = real)
      @test S1 == S2
      @test A1 == A2
      ref = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      t = ET.sparse_equivariant_tensor_spec(Val(0); mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      @test Matrix(t.symm) == Matrix(ref.A2Bmaps[1])
      @test t.𝔸spec == ref.meta["𝔸spec"]
      @test t.Aspec == ref.meta["Aspec"]
      @test collect(t.Aspec_raw) == collect(ref.abasis.spec)
   end
   # complex basis and L = 1 go through the same table
   mb = trim_mb_spec(1, 2, 3)
   @test ET.symmetrisation_matrix(Val(0), mb; PI = true, basis = complex) ==
         ET.symmetrisation_matrix(0, mb; PI = true, basis = complex)
   @test ET.symmetrisation_matrix(Val(1), mb; PI = true, basis = real) ==
         ET.symmetrisation_matrix(1, mb; PI = true, basis = real)
   # the static table is bounded
   mb9 = [[(n = 1, l = 0) for _ in 1:9]]
   @test_throws ArgumentError ET.symmetrisation_matrix(Val(0), mb9; PI = true, basis = real)
   @test_throws ArgumentError ET.O3.coupling_coeffs(Val(0), [0, 0], [1]; PI = true, basis = real)
end
