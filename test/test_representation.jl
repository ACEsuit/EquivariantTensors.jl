using Test, EquivariantTensors, StaticArrays, SpheriCart, Combinatorics
using EquivariantTensors.O3: Ctran
using PartialWaveFunctions: clebschgordan
using LinearAlgebra
using WignerD, Rotations

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")


##

@info("Testing the correctness of Ctran(L)")
Lmax = 4
for L = 0:Lmax
   @info("""Testing whether or not we found a correct transformation between 
            cSH to rSH for L = $L""")
   for ntest = 1:30
      x = @SVector rand(3)
      Ylm = cYlm(L,x)[L^2+1:(L+1)^2]
      Ylm_r = rYlm(L,x)[L^2+1:(L+1)^2]
      print_tf(@test norm(Ctran(L)' * Ylm_r - collect(Ylm)) < 1e-12)
   end
   println()
end

##

@info("Testing the D-matrix for cSH")
Lmax = 4
for ntest = 1:30
   local θ, Q
   x = @SVector rand(3)
   θ = rand(3) * 2pi
   Q = RotZYZ(θ...)
   Ylm = cYlm(Lmax,x;method=:LC)
   Ylm_r = cYlm(Lmax,Q * x;method=:LC)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = transpose(WignerD.wignerD(L, θ...)) 
      # Here transpose is needed because of the different convention (between 
      # ours and WignerD.jl's) of the D-matrix; 
      # Same for the cases below.
      print_tf(@test norm(D * YrL- YL) < 1e-12)
   end
end
println()

##

@info("The cSH D-matrix recursion")
Lmax = 4
θ = rand(3) * 2pi
Dset = [ transpose(WignerD.wignerD(L, θ...)) for L = 0:Lmax ]

for ntest = 1:100
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)

   val1 = Dset[l1+1][m1+l1+1,μ1+l1+1] * Dset[l2+1][m2+l2+1,μ2+l2+1]
   val2 = 0
   for λ = abs(l1-l2):l1+l2
      if -λ ≤ m1+m2 ≤ λ && -λ ≤ μ1+μ2 ≤ λ
         val2 += ( clebschgordan(l1,m1,l2,m2,λ,m1+m2) 
                   * clebschgordan(l1,μ1,l2,μ2,λ,μ1+μ2) 
                   * Dset[λ+1][m1+m2+λ+1,μ1+μ2+λ+1] )
      end
   end
   print_tf(@test norm(val1 - val2) < 1e-12)
end
println()

##

@info("Testing the D-matrix for rSH")
Lmax = 4
for ntest = 1:30
   local θ, Q
   x = @SVector rand(3)
   θ = rand(3) * 2pi
   Q = RotZYZ(θ...)
   Ylm = rYlm(Lmax, x)
   Ylm_r = rYlm(Lmax, Q * x)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = transpose(WignerD.wignerD(L, θ...))
      D_real = Ctran(L) * D * Ctran(L)'
      print_tf(@test norm(D_real * YrL - YL) < 1e-12)
   end
end
println()

##

## NOTE: In contradict to my intuition, the recursion for rSH D-matrix is not 
#        the same as the cSH one. The correct recursion relies not only on a 
#        single previous value but 16 (or 8, if we investigate the property 
#        of the new D matrix). Here comes the correct recursion!

@info("The rSH D-matrix recursion")
Lmax = 4
θ = rand(3) * 2pi
Dset = [ Ctran(L) * transpose(WignerD.wignerD(L, θ...)) * Ctran(L)' for L = 0:Lmax ]
DDset = [ transpose(WignerD.wignerD(L, θ...)) for L = 0:Lmax ]

for ntest = 1:30
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)

   val1 = Dset[l1+1][m1+l1+1,μ1+l1+1] * Dset[l2+1][m2+l2+1,μ2+l2+1]
   val2 = 0
   for λ = abs(l1-l2):l1+l2, m in unique([m1,-m1]), mm in unique([m2,-m2])
      for μ in unique([μ1,-μ1]), μμ in unique([μ2,-μ2])
         if abs(m+mm) ≤ λ && abs(μ+μμ) ≤ λ
            c1 = Ctran(m1,m) * Ctran(m2,mm) * Ctran(μ1,μ)' * Ctran(μ2,μμ)'
            c2 = clebschgordan(l1,m,l2,mm,λ,m+mm) * clebschgordan(l1,μ,l2,μμ,λ,μ+μμ)
            # val2 += c1 * DDset[l1+1][m+l1+1,μ+l1+1] * DDset[l2+1][mm+l2+1,μμ+l2+1]
            # val2 += c1 * c2 * DDset[λ+1][m+mm+λ+1,μ+μμ+λ+1]
            for p in unique([m+mm,-m-mm]), q in unique([μ+μμ,-μ-μμ])
               c3 = Ctran(p,m+mm)' * Ctran(q,μ+μμ)
               val2 += c1 * c2 * c3 * Dset[λ+1][p+λ+1,q+λ+1]
            end
         end
      end
   end
   print_tf(@test norm(val1 - val2) < 1e-12)
end
println()
