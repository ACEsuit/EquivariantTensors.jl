using Test, EquivariantTensors, StaticArrays, LinearAlgebra
using EquivariantTensors.O3: coupling_coeffs

# nullspace_solver = :dense must span the same coupled space as the default
# (:sparse, UMFPACK); single basis functions may differ by a sign, and within a
# degenerate (nn, ll) block by an orthogonal rotation, so compare row spaces.

_flat(U::AbstractMatrix{<:Number}) = Matrix(U)
_flat(U::AbstractMatrix{<:SVector}) = [U[i, j][k] for i = 1:size(U, 1), j = 1:size(U, 2), k = 1:length(U[1])] |>
                                       X -> reshape(X, size(U, 1), :)
_proj(X) = pinv(X) * X                          # orthogonal projector onto the row space

cases = [ (SA[1, 1], SA[1, 1]), (SA[1, 1, 1], SA[1, 1, 2]), (SA[1, 1, 1], SA[2, 2, 2]),
          (SA[1, 2, 1], SA[1, 1, 2]), (SA[1, 1, 1, 1], SA[1, 1, 1, 1]), (SA[1, 1, 2, 2], SA[1, 1, 2, 2]),
          (SA[1, 1, 1, 1], SA[2, 2, 2, 2]), (SA[1, 1, 1, 1], SA[1, 2, 2, 3]) ]

@testset "nullspace_solver = :dense spans the :sparse coupled space" begin
   ndiff = 0
   for L = 0:2, basis in (complex, real), (nn, ll) in cases
      Us, MMs = coupling_coeffs(L, ll, nn; PI = true, basis = basis)
      Ud, MMd = coupling_coeffs(L, ll, nn; PI = true, basis = basis, nullspace_solver = :dense)
      @test MMs == MMd
      @test size(Us) == size(Ud)
      isempty(Us) && continue
      Xs, Xd = _flat(Us), _flat(Ud)
      @test rank(Xd) == rank(Xs) == size(Us, 1)
      @test norm(_proj(Xs) - _proj(Xd)) < 1e-10
      ndiff += !(Xs ≈ Xd)
   end
   @info("nullspace_solver: $ndiff of the coupling sets differ elementwise (same row space)")
end

@testset "nullspace_solver default and validation" begin
   ll, nn = SA[1, 1, 2], SA[1, 1, 1]
   @test coupling_coeffs(0, ll, nn; basis = real) == coupling_coeffs(0, ll, nn; basis = real, nullspace_solver = :sparse)
   @test_throws ErrorException coupling_coeffs(0, ll, nn; nullspace_solver = :qr)
end

@testset "symmetrisation_matrix with nullspace_solver = :dense" begin
   mb_spec = [ [(n = n1, l = l1), (n = n2, l = l2), (n = n3, l = l3)]
               for n1 = 1:2 for n2 = n1:2 for n3 = n2:2 for l1 = 0:2 for l2 = 0:2 for l3 = 0:2
               if (l1 + l2 + l3) <= 4 && issorted([(n1, l1), (n2, l2), (n3, l3)]) ]
   for L = 0:1
      As, specs = EquivariantTensors.symmetrisation_matrix(L, mb_spec; prune = true, PI = true, basis = real)
      Ad, specd = EquivariantTensors.symmetrisation_matrix(L, mb_spec; prune = true, PI = true, basis = real,
                                                           nullspace_solver = :dense)
      @test specs == specd
      @test size(As) == size(Ad)
      Xs, Xd = _flat(Matrix(As)), _flat(Matrix(Ad))
      @test norm(_proj(Xs) - _proj(Xd)) < 1e-10
   end
end

@testset "Val(L) API forwards nullspace_solver" begin
   mb_spec = [ [(n = 1, l = 1), (n = 1, l = 1), (n = 2, l = 2)], [(n = 1, l = 2), (n = 1, l = 2), (n = 1, l = 2), (n = 1, l = 2)] ]
   for solver in (:sparse, :dense)
      A, spec = EquivariantTensors.symmetrisation_matrix(0, mb_spec; prune = true, PI = true, basis = real,
                                                         nullspace_solver = solver)
      Av, specv = EquivariantTensors.symmetrisation_matrix(Val(0), mb_spec; prune = true, PI = true, basis = real,
                                                           nullspace_solver = solver)
      @test spec == specv && A == Av
   end
   @test_throws ArgumentError EquivariantTensors.O3.coupling_coeffs(Val(0), [1, 1], [1, 1]; nullspace_solver = :qr)
end
