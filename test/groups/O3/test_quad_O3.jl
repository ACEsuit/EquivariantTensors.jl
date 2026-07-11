
import EquivariantTensors: O3
using Rotations, WignerD, StaticArrays, LinearAlgebra, Test
using ACEbase.Testing: print_tf

# ===================== QuadSO3 tests =====================

##

@info("Test QuadSO3: weights sum to 1 and q(1) = 1")
for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
   q = O3.QuadSO3(N)
   print_tf(@test sum(w for (w, _) in q) ≈ 1.0)
   print_tf(@test q(R -> 1.0) ≈ 1.0)
end
println()

##

@info("Test QuadSO3: Wigner-D orthogonality ∫ Dˡₘₘ′(R) dR = δ_{l,0} δ_{m,0} δ_{m′,0}")
# A degree-N rule should integrate all Wigner-D elements exactly up to l = N.
# D⁰₀₀ = 1, so its integral is 1.
# For l ≥ 1, all matrix elements of Dˡ should integrate to 0.

for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   q = O3.QuadSO3(N)
   for l in 0:q.degree
      val = q(R -> begin
         θ = RotZYZ(R)
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

@info("Test QuadSO3: degree-N rule is NOT exact at l = N+1")
for N in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   q = O3.QuadSO3(N)
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

##

@info("Test QuadSO3: error for unavailable degree")
print_tf(@test_throws ErrorException O3.QuadSO3(15))
println()

# ===================== QuadO3 tests =====================

##

@info("Test QuadO3: weights sum to 1 and q(1) = 1")
for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
   q = O3.QuadO3(N)
   print_tf(@test sum(w for (w, _) in q) ≈ 1.0)
   print_tf(@test q(R -> 1.0) ≈ 1.0)
end
println()

##

@info("Test QuadO3: nodes include proper rotations and improper rotations")
for N in [1, 5, 14]
   q = O3.QuadO3(N)
   dets = [round(det(R)) for (_, R) in q]
   n_proper = count(d -> d ≈ 1.0, dets)
   n_improper = count(d -> d ≈ -1.0, dets)
   print_tf(@test n_proper == n_improper == length(q) ÷ 2)
end
println()

##

@info("Test QuadO3: O(3) orthogonality for Wigner-D")
# On O(3), the irreps are labelled by (l, p) where p = ±1 is the parity.
# Under inversion (det = -1), Dˡ(Q) picks up a factor of det(Q)^l = (-1)^l
# for the standard irrep, or det(Q)^l * p for general parity.
#
# The O(3) Haar integral of Dˡₘₘ′(Q) is:
#   ∫_{O(3)} Dˡ(Q) dQ = ½[∫_{SO(3)} Dˡ(R) dR + (-1)^l ∫_{SO(3)} Dˡ(R) dR]
# which equals δ_{l,0} for even l and 0 for odd l.
#
# More generally, for f(Q) = det(Q)^p * Dˡ(Q):
#   ∫_{O(3)} det(Q)^p Dˡ(Q) dQ = δ_{l,0} δ_{p even}

for N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   q = O3.QuadO3(N)
   for l in 0:q.degree
      # Test ∫ Dˡ(Q) dQ (should be δ_{l,0})
      val = q(Q -> begin
         R = det(Q) < 0 ? -Q : Q
         θ = RotZYZ(SMatrix{3,3}(R))
         det(Q)^l * WignerD.wignerD(l, θ.theta1, θ.theta2, θ.theta3)
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

@info("Test QuadO3: parity-odd functions integrate to zero")
# ∫_{O(3)} det(Q) * Dˡ(Q) dQ = 0 for all l (since det(Q) is parity-odd)
for N in [1, 3, 5, 9, 14]
   q = O3.QuadO3(N)
   for l in 0:q.degree
      val = q(Q -> begin
         R = det(Q) < 0 ? -Q : Q
         θ = RotZYZ(SMatrix{3,3}(R))
         det(Q)^(l+1) * WignerD.wignerD(l, θ.theta1, θ.theta2, θ.theta3)
      end)
      print_tf(@test norm(val) < 1e-10)
   end
end
println()
