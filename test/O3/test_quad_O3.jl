
import EquivariantTensors: O3
using Rotations, WignerD, StaticArrays, LinearAlgebra, Test
using ACEbase.Testing: print_tf

##

@info("Test QuadO3: weights sum to 1 and q(1) = 1")
for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
   q = O3.QuadO3(N)
   print_tf(@test sum(w for (w, _) in q) ≈ 1.0)
   print_tf(@test q(R -> 1.0) ≈ 1.0)
end
println()

##

@info("Test QuadO3: Wigner-D orthogonality ∫ Dˡₘₘ′(R) dR = δ_{l,0} δ_{m,0} δ_{m′,0}")
# A degree-N rule should integrate all Wigner-D elements exactly up to l = N.
# D⁰₀₀ = 1, so its integral is 1.
# For l ≥ 1, all matrix elements of Dˡ should integrate to 0.

for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
   q = O3.QuadO3(N)
   for l in 0:N
      val = q(R -> begin
         θ = RotZYZ(R)
         # D_from_angles returns conj(wignerD) so that y∘Q = D*y.
         # For the orthogonality test we use the raw wignerD matrix elements.
         WignerD.wignerD(l, θ.theta1, θ.theta2, θ.theta3)
      end)
      if l == 0
         print_tf(@test val[1,1] ≈ 1.0)
      else
         print_tf(@test norm(val) < 1e-10)
      end
   end
end
println()

##

@info("Test QuadO3: degree-N rule is NOT exact at l = N+1")
# Verify that we don't accidentally have higher-order accuracy.
# At least one element of ∫ Dˡ dR should be noticeably non-zero for l = N+1.
# (This isn't guaranteed for every single rule due to symmetry, but should hold
# for most. We check a weaker condition: the error is above 1e-14.)

for N in [2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
   q = O3.QuadO3(N)
   # test at l = N+1 and l = N+2 — at least one should show non-zero integral
   maxerr = 0.0
   for l in (N+1):(N+2)
      val = q(R -> begin
         θ = RotZYZ(R)
         WignerD.wignerD(l, θ.theta1, θ.theta2, θ.theta3)
      end)
      maxerr = max(maxerr, norm(val))
   end
   print_tf(@test maxerr > 1e-14)
end
println()
