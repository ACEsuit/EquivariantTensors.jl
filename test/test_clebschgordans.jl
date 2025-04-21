
import PartialWaveFunctions as PWF 
using Random

Random.seed!(1234)

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

##

# our original reference implementation of CG coeffs, which we no longer 
# use, but we keep this test to make sure our assumptions are consistent 
# with the PartialWaveFunctions implementation. 
module _CGold

cg_conditions(j1,m1, j2,m2, J,M) =
	cg_l_condition(j1, j2, J)   &&
	cg_m_condition(m1, m2, M)   &&
	(abs(m1) <= j1) && (abs(m2) <= j2) && (abs(M) <= J)

cg_l_condition(j1, j2, J) = (abs(j1-j2) <= J <= j1 + j2)

cg_m_condition(m1, m2, M) = (M == m1 + m2)


"""
`clebschgordan(j1, m1, j2, m2, J, M, T=Float64)` :
A reference implementation of Clebsch-Gordon coefficients based on
https://hal.inria.fr/hal-01851097/document
Equation (4-6)
This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks, but only precomputation.
The ordering of parameters corresponds to the following convention:
```
clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
```
where
```
   D_{m1k1}^{l1} D_{m2k2}^{l2}}
	=
	∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
```
"""
function clebschgordan(j1, m1, j2, m2, J, M, T=Float64)
	if !cg_conditions(j1, m1, j2, m2, J, M)
		return zero(T)
	end

   N = (2*J+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(J+M)) * factorial(big(J-M)) /
       factorial(big( j1+j2-J)) /
       factorial(big( j1-j2+J)) /
       factorial(big(-j1+j2+J)) /
       factorial(big(j1+j2+J+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2-J-m1, j1-J+m2)
   ub = (j1+j2-J, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      bk = big(k)
      G += (-1)^k *
           binomial(big( j1+j2-J), big(k)) *
           binomial(big( j1-j2+J), big(j1-m1-k)) *
           binomial(big(-j1+j2+J), big(j2+m2-k))
   end

   return T(sqrt(N) * G)
end

end

##

@info("Testing the correctness of PartialWaveFunctions with our assumptions")

Lmax = 6
TOL = 1e-14
nerr = 0

for j1 in 1:Lmax, m1 in -j1:j1
    for j2 in 1:Lmax, m2 in -j2:j2
        for J in 1:Lmax, M in -J:J
            global nerr  
            C1 = _CGold.clebschgordan(j1, m1, j2, m2, J, M)
            C2 = PWF.clebschgordan(j1, m1, j2, m2, J, M)
            if abs(C1 - C2) > TOL 
                @error("""Clebsch-Gordan error: 
                                j1 = $j1, m1 = $m1, 
                                j2 = $j2, m2 = $m2, 
                                J = $J, M = $M""")
                nerr += 1
            end
        end
    end
end

if nerr > 0 
    @error("There are $nerr errors in the Clebsch-Gordan coefficients test")
else
    @info("All tested Clebsch-Gordan coefficients are correct")
end

@test nerr == 0

## 

# The following is an alternative test of Clebsch Gordans that should 
# technically be enough on its own. (we keep both for now...)

@info("Checking the SphH expansion in terms of CG coeffs")
# expansion coefficients of a product of two spherical harmonics in terms a
# single spherical harmonic
# see e.g. https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients
# this is the magic formula that we need, on which everything else is based

Lmax = 10
TOL = 1e-14
NTEST = 30 

for _ = 1:NTEST 
      local θ, M 
      # two random Ylm  ...
      l1, l2 = rand(1:10), rand(1:10)
      m1, m2 = rand(-l1:l1), rand(-l2:l2)
      # ... evaluated at random spherical coordinates
      θ = rand() * π
      local φ = (rand()-0.5) * 2*π
      R = SVector( cos(φ)*sin(θ), sin(φ)*sin(θ), cos(θ) )
      # evaluate all relevant Ylms (up to l1 + l2)
      
      Ylm = cYlm(l1 + l2, R)
      # evaluate the product p = Y_l1_m1 * Y_l2_m2
      p = Ylm[lm2idx(l1,  m1)] * Ylm[lm2idx(l2,m2)]
      # and its expansion in terms of CG coeffs
      p2 = 0.0
      M = m1 + m2  # all other coeffs are zero

      for L = abs(M):(l1+l2)
            p2 += sqrt( (2*l1+1)*(2*l2+1) / (4 * π * (2*L+1)) ) *
                PWF.clebschgordan(l1,  0, l2,  0, L, 0) *
                PWF.clebschgordan(l1, m1, l2, m2, L, M) *
                Ylm[lm2idx(L, M)]
      end
      print_tf((@test (p ≈ p2) || (abs(p-p2) < 1e-14)))
end
println()
