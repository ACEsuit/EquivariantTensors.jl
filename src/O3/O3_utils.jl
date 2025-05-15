using PartialWaveFunctions, SparseArrays

import EquivariantTensors.SO3: Ctran, Q_from_angles, D_from_angles, QD_from_angles
# -----------------------------------------------------------------
#  complex and real Clebsch-Gordan coefficients

function cg(l1, m1, l2, m2, L, M, basis::typeof(complex))
   return PartialWaveFunctions.clebschgordan(l1, m1, l2, m2, L, M)
end

function cg(l1, m1, l2, m2, L, M, basis::typeof(real))
   return _real_clebschgordan(l1, m1, l2, m2, L, M) 
end
 
function _real_clebschgordan(l1, m1, l2, m2, λ, νp)
   result = 0.0 + 0im

   # Only n values with |n| = |m| contribute due to Ctran(m, n) selection rule
   n1_values = m1 == 0 ? [0] : [-m1, m1]
   n2_values = m2 == 0 ? [0] : [-m2, m2]

   for n1 in n1_values
       for n2 in n2_values
           ν = n1 + n2
           # Selection rules: |ν| must be ≤ λ and match |νp|
           if abs(ν) ≤ λ && abs(ν) == abs(νp)
               cg = PartialWaveFunctions.clebschgordan(l1, n1, l2, n2, λ, ν)
               result += Ctran(m1, n1) * conj(Ctran(-m2, -n2)) * conj(Ctran(νp, ν)) * cg
           end
       end
   end

   return real(result)
end
 

function cgmatrix(l1, l2, λ; basis = real)
   cgm = zeros((2l1+1)*(2l2+1), 2λ+1)
   for (i, (m1, m2)) in enumerate(Iterators.product(-l1:l1, l2:-1:-l2))
      for ν = -λ : λ
         if abs(ν) <= λ
            cgm[i, ν + λ + 1] = (-1)^m2 * cg(l1, m1, l2, m2, λ, ν, basis)
         end
      end
   end
   return sparse(cgm) 
end