using StaticArrays, LinearAlgebra, RepLieGroups, WignerD, Rotations
using Test

include("../src/utils/utils_for_tests.jl")

for ntest = 1:200
   ll = SA[rand(0:1, 6)...] |> sort
   nn = SA[rand(1:2, 6)...] |> sort
   # nn = SA[1,1,1,1,2,2]
   # ll = SA[1,2,3,1,2,3]
   N1 = rand(1:length(ll)-1) # random partition
   
   Ltot = rand(0:4)
   if isodd(sum(ll)+Ltot); continue; end
   # if isodd(sum(ll)+Ltot); Ltot += 1; end
   t1 = @elapsed C_re_semi_pi, MM = re_semi_pi(nn,ll,Ltot,N1)
   t2 = @elapsed C_re,_,_,M = re_rpe(nn,ll,Ltot)
   t3 = @elapsed C_rpe,M = rpe_basis_new(nn,ll,Ltot)
   @show t1, t2, t3
   
   if rank(gram(C_rpe)) > 0
      # @info("Test that re_semi_pi span a set with rank ranging between RE and RPE")
      @test rank(gram(C_rpe)) <= rank(gram(C_re_semi_pi)) <= rank(gram(C_re))

      # @info("Testing the equivariance of the old RPE basis")
      Rs = rand_config(length(ll))
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      D = transpose(WignerD.wignerD(Ltot, θ...)) 
      QRs = [Q*Rs[i] for i in 1:length(Rs)]
      fRs1 = eval_basis(Rs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
      fRs1Q = eval_basis(QRs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
      Ltot == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14)

      # @info("Test that re_semi_pi span a larger space than RPE")
      # Do the rand batch on the same set of points
      ntest = 1000
      ORD = length(ll) # length of each group 
      BB1 = complex.(zeros(typeof(C_re_semi_pi[1]), size(C_re_semi_pi, 1), ntest))
      BB2 = complex.(zeros(typeof(C_rpe[1]), size(C_rpe, 1), ntest))
      for i = 1:ntest 
         # construct a random set of particles with 𝐫 ∈ ball(radius=1)
         Rs = [ rand_ball() for _ in 1:ORD ]
         BB1[:, i] = eval_basis(Rs; coeffs=C_re_semi_pi, MM=MM, ll=ll, nn=nn) 
         BB2[:, i] = eval_basis(Rs; coeffs=C_rpe, MM=MM, ll=ll, nn=nn) 
      end
      @test rank(gram(C_re_semi_pi)) == rank(gram(BB1); rtol=1e-11) == rank(gram([BB1;BB2]); rtol=1e-11) >= rank(gram(BB2); rtol=1e-11) == rank(gram(C_rpe))
   end
end

# When the partition gives non-intersect nn's and ll's

# llset = [SA[1,1,1,1], SA[1,1,2,2], SA[1,2,2,2], SA[1,1,2,2,2], SA[1,1,2,2,3], SA[1,1,1,2,2,3], SA[1,1,1,1,2,2,3], SA[1,1,1,1,2,2,2,2] ]
# nnset = [SA[1,1,2,2], SA[1,1,2,3], SA[1,1,2,3], SA[1,2,1,2,2], SA[1,1,1,1,1], SA[1,2,2,1,2,1], SA[1,1,1,2,1,2,1], SA[1,1,2,2,1,1,1,1] ]
# Partition = [2,2,1,2,2,3,4,4]

# To test if we gain efficiency for larger correlation order - for this one, the old RPE basis is much slower
# llset = [SA[1,1,1,1,1,2,2,2,2,2]]
# nnset = [SA[1,1,1,1,1,1,1,1,1,1]]
# Partition = [5]

# To test the equivariance violation for large sum(ll_i)
# llset = [SA[5,5,4,4]] |> sort
# nnset = [SA[1,1,1,1]]
# Partition = [2]

using Combinatorics
lmax = 4
nmax = 4
nnll_list = [] 
for ORD = 2:6
   for ll in with_replacement_combinations(1:lmax, ORD) 
      # 0 or 1 above ?
      # if !iseven(sum(ll)+L); continue; end  # This is to ensure the reflection symmetry
      if sum(ll) > 3 * lmax; continue; end 
      for Inn in CartesianIndices( ntuple(_->1:nmax, ORD) )
         nn = [ Inn.I[α] for α = 1:ORD ]
         if sum(nn) > sum(1:nmax); continue; end
         nnll = [ (ll[α], nn[α]) for α = 1:ORD ]
         if !issorted(nnll); continue; end
         push!(nnll_list, (SVector(nn...), SVector(ll...)))
      end
   end
end

nnll_list_short = nnll_list[1:100:end]

for i = 1:length(nnll_list_short)
   ll = nnll_list_short[i][2]
   nn = nnll_list_short[i][1]
   # Partitionset = RepLieGroups.Sn(nn,ll)
   # if length(Partitionset) <= 2; continue; end
   # N1 = Partitionset[rand(2:length(Partitionset)-1)]-1
   N1 = rand(1:length(ll)-1) # Instead of fine partition which gives two non-intersecting sets, we now allow random partition

   for Ltot in (iseven(sum(ll)) ? (0:2:4) : (1:2:3))
      println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
      println()
      t_re_semi_pi = @elapsed C_re_semi_pi, MM = re_semi_pi(nn,ll,Ltot,N1) # no longer needed and has been tested above - but shown here to see how long does the last symmetrization take
      t_rpe = @elapsed C_rpe,M = rpe_basis_new(nn,ll,Ltot)
      t_recursive = @elapsed C_rpe_recursive, MM = rpe_basis_new(nn,ll,Ltot,N1; symmetrization_method = :explicit)
      t_recursive_2 = @elapsed C_rpe_recursive_kernel, MM_2 = rpe_basis_new(nn,ll,Ltot,N1; symmetrization_method = :kernel)
      
      # make sure the order of the basis is the same
      if size(C_rpe_recursive,1) == size(C_rpe,1) == size(C_rpe_recursive_kernel,1) != 0
         if MM != M || MM != MM_2
            @assert sort(MM) == sort(M) == sort(MM_2)
            ord = sortperm(MM)
            @assert MM[ord] = sort(MM)
            C_rpe_recursive = C_rpe_recursive[:,ord]
            ord = sortperm(M)
            @assert M[ord] = sort(M)
            C_rpe = C_rpe[:,ord]
            ord = sortperm(MM_2)
            @assert MM_2[ord] = sort(MM_2)
            C_rpe_recursive_kernel = C_rpe_recursive_kernel[:,ord]
            MM = sort(MM)
         end
      end

      @show t_re_semi_pi, t_recursive, t_recursive_2, t_rpe
      println()

      if rank(gram(C_rpe)) > 0
         # @info("Test that re_semi_pi span a set with rank ranging between RE and RPE")
         # @test rank(gram(C_rpe)) == rank(gram(C_rpe_recursive)) == rank(gram(C_rpe_recursive_kernel)) == rank(gram([C_rpe;C_rpe_recursive;C_rpe_recursive_kernel]))
         # In fact, it would be more interesting to check the following, but it makes less sense than the above test (not as intuitive)
         # This is because we already tested elsewhere that C_rpe has full rank hence the above is sufficient to show the equivalence
         @test size(C_rpe,1) == size(C_rpe_recursive,1) == size(C_rpe_recursive_kernel,1) == rank(gram([C_rpe;C_rpe_recursive;C_rpe_recursive_kernel]))

         # @info("Testing the equivariance of the old RPE basis")
         Rs = rand_config(length(ll))
         θ = rand(3) * 2pi
         Q = RotZYZ(θ...)
         D = transpose(WignerD.wignerD(Ltot, θ...)) 
         QRs = [Q*Rs[i] for i in 1:length(Rs)]
         # fRs1 = eval_basis(Rs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
         # fRs1Q = eval_basis(QRs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
         fRs1 = eval_basis(Rs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         fRs1Q = eval_basis(QRs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         Ltot == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-12) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-12)
         fRs1 = eval_basis(Rs; coeffs = C_rpe_recursive_kernel, MM = MM, ll = ll, nn = nn)
         fRs1Q = eval_basis(QRs; coeffs = C_rpe_recursive_kernel, MM = MM, ll = ll, nn = nn)
         Ltot == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-12) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-12)

         # @info("Test that re_semi_pi span the same space as RPE")
         # Do the rand batch on the same set of points
         ntest = 1000
         ORD = length(ll) # length of each group 
         BB1 = complex.(zeros(typeof(C_rpe_recursive[1]), size(C_rpe_recursive, 1), ntest))
         BB2 = complex.(zeros(typeof(C_rpe[1]), size(C_rpe, 1), ntest))
         BB3 = complex.(zeros(typeof(C_rpe_recursive_kernel[1]), size(C_rpe_recursive_kernel, 1), ntest))
         for i = 1:ntest 
            # construct a random set of particles with 𝐫 ∈ ball(radius=1)
            Rs = [ rand_ball() for _ in 1:ORD ]
            BB1[:, i] = eval_basis(Rs; coeffs=C_rpe_recursive, MM=MM, ll=ll, nn=nn)
            BB2[:, i] = eval_basis(Rs; coeffs=C_rpe, MM=MM, ll=ll, nn=nn) 
            BB3[:, i] = eval_basis(Rs; coeffs=C_rpe_recursive_kernel, MM=MM, ll=ll, nn=nn)
         end
         @test rank(gram(BB1); rtol=1e-11) == rank(gram(BB2); rtol=1e-11) == rank(gram([BB1;BB2;BB3]); rtol=1e-11) == size(C_rpe,1)
      end
   end
end

# The last test is to show the efficiency of the new recursive method
# which can also be an example showing that how much we may get if we 
# store in advance some coepleing coefficients.

for N = 6:12
   nn = SA[ones(Int64,N)...] .* rand(1:5)
   ll = SA[ones(Int64,N)...]
   N1 = Int(round(N/2))

   for Ltot in (iseven(sum(ll)) ? (0:2:4) : (1:2:3))
      t_rpe = @elapsed C_rpe, M = RepLieGroups.rpe_basis_new(nn,ll,Ltot) # reference time
      t_re_semi_pi = @elapsed C_re_semi_pi, MM = re_semi_pi(nn,ll,Ltot,N1) # time for re_semi_pi - which can be avoided by storing the coupling coefficients
      t_rpe_recursive_kernel = @elapsed C_rpe_recursive, MM = RepLieGroups.rpe_basis_new(nn,ll,Ltot,N1; symmetrization_method = :kernel) # time for rpe_basis_new with kernel symmetrization - the difference to the above should be the time for symmetrization

      println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
      println("Standard RPE basis : $t_rpe")
      println("RE_SEMI_PI basis : $t_re_semi_pi")
      println("Recursive RPE basis : $t_rpe_recursive_kernel")
      println()

      if size(C_rpe_recursive,1) == size(C_rpe,1) != 0
         if MM != M
            @assert sort(MM) == sort(M)
            ord = sortperm(MM)
            @assert MM[ord] = sort(MM)
            C_rpe_recursive = C_rpe_recursive[:,ord]
            ord = sortperm(M)
            @assert M[ord] = sort(M)
            C_rpe = C_rpe[:,ord]
            MM = sort(MM)
         end
      end

      @test size(C_rpe,1) == size(C_rpe_recursive,1) == rank(gram([C_rpe;C_rpe_recursive]), rtol=1e-11)
   end
end