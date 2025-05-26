using StaticArrays, LinearAlgebra, EquivariantTensors, WignerD, Rotations, Combinatorics
using Test, Random
using EquivariantTensors.O3: gram, coupling_coeffs

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

println()
lmax = 4
nmax = 4
nnll_list = [] 
for ORD = 2:6
   for ll in with_replacement_combinations(1:lmax, ORD) 
      # 0 or 1 above ?
      # if !iseven(sum(ll)+L); continue; end  # This is to ensure the reflection symmetry
      if sum(ll) > 3 * lmax; continue; end # avoid too large sum(ll) which may slow down the test
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

@info("Test the recursive RPE basis - ordered RPE basis")
for i = 1:length(nnll_list_short)
   ll = nnll_list_short[i][2]
   nn = nnll_list_short[i][1]

   k = rand(1:length(ll)-1) # we wanted to split nn and ll into k+1 blocks
   N1 = sort(shuffle(collect(1:length(ll)-1))[1:k]) # position where we split

   for Ltot in (iseven(sum(ll)) ? (0:2:4) : (1:2:3))
      println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
      
      # Two different ways to construct the RPE basis
      C_rpe,M = coupling_coeffs(Ltot,ll,nn)
      C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn,N1)
      
      # make sure the order of the basis is the same
      if size(C_rpe_recursive,1) == size(C_rpe,1) != 0
         if MM != M
            @assert sort(MM) == sort(M)
            ord = sortperm(MM)
            @assert MM[ord] = sort(MM)
            C_rpe_recursive = C_rpe_recursive[:,ord]
            ord = sortperm(M)
            @assert M[ord] = sort(M)
            C_rpe = C_rpe[:,ord]
         end
      end

      if rank(gram(C_rpe)) > 0
         # @info("Test that re_semi_pi span a set with rank ranging between RE and RPE")
         # @test rank(gram(C_rpe)) == rank(gram(C_rpe_recursive)) == rank(gram([C_rpe;C_rpe_recursive]))
         # In fact, it would be more interesting to check the following, but it makes less sense than the above test (not as intuitive)
         # This is because we already tested elsewhere that C_rpe has full rank hence the above is sufficient to show the equivalence
         print_tf(@test size(C_rpe,1) == size(C_rpe_recursive,1) == rank(gram([C_rpe;C_rpe_recursive])))

         # @info("Testing the equivariance of the two recursive RPE basis")
         local Rs = rand_config(length(ll))
         local θ = rand(3) * 2pi
         local Q = RotZYZ(θ...)
         local D = transpose(WignerD.wignerD(Ltot, θ...)) 
         local QRs = [Q*Rs[i] for i in 1:length(Rs)]

         fRs1 = eval_sym_basis(Rs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         fRs1Q = eval_sym_basis(QRs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         Ltot == 0 ? (print_tf(@test norm(fRs1 - fRs1Q) < 1e-12)) : (print_tf(@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-12))


         # @info("Test that the two recursive RPEs span the same space as RPE")
         # Do the rand batch on the same set of points
         ntest = 1000
         ORD = length(ll) # length of each group 
         BB1 = complex.(zeros(typeof(C_rpe_recursive[1]), size(C_rpe_recursive, 1), ntest))
         BB2 = complex.(zeros(typeof(C_rpe[1]), size(C_rpe, 1), ntest))
         for i = 1:ntest 
            # construct a random set of particles with 𝐫 ∈ ball(radius=1)
            Rs = [ rand_ball() for _ in 1:ORD ]
            BB1[:, i] = eval_sym_basis(Rs; coeffs=C_rpe_recursive, MM=MM, ll=ll, nn=nn)
            BB2[:, i] = eval_sym_basis(Rs; coeffs=C_rpe, MM=MM, ll=ll, nn=nn) 
         end
         print_tf(@test rank(gram(BB1); rtol=1e-11) == rank(gram(BB2); rtol=1e-11) == rank(gram([BB1;BB2]); rtol=1e-11) == size(C_rpe,1))
      end
      println()
   end
end

@info("Test the recursive RPE basis - unordered RPE basis")
for i = 1:length(nnll_list_short)
   ll = nnll_list_short[i][2]
   nn = nnll_list_short[i][1]

   k = rand(1:length(ll)-1) # we wanted to split nn and ll into k+1 blocks
   N1 = sort(shuffle(collect(1:length(ll)-1))[1:k]) # position where we split

   for Ltot in (iseven(sum(ll)) ? (0:2:4) : (1:2:3))
      println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
      
      # Two different ways to construct the RPE basis
      C_rpe,M = coupling_coeffs(Ltot,ll,nn; ordered = false)
      C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn,N1; ordered = false)
      
      # make sure the order of the basis is the same
      if size(C_rpe_recursive,1) == size(C_rpe,1) != 0
         if MM != M
            @assert sort(MM) == sort(M)
            ord = sortperm(MM)
            @assert MM[ord] = sort(MM)
            C_rpe_recursive = C_rpe_recursive[:,ord]
            ord = sortperm(M)
            @assert M[ord] = sort(M)
            C_rpe = C_rpe[:,ord]
         end
      end

      if rank(gram(C_rpe)) > 0
         # @info("Test that re_semi_pi span a set with rank ranging between RE and RPE")
         # @test rank(gram(C_rpe)) == rank(gram(C_rpe_recursive)) == rank(gram([C_rpe;C_rpe_recursive]))
         # In fact, it would be more interesting to check the following, but it makes less sense than the above test (not as intuitive)
         # This is because we already tested elsewhere that C_rpe has full rank hence the above is sufficient to show the equivalence
         print_tf(@test size(C_rpe,1) == size(C_rpe_recursive,1) == rank(gram([C_rpe;C_rpe_recursive])))

         # @info("Testing the equivariance of the two recursive RPE basis")
         local Rs = rand_config(length(ll))
         local θ = rand(3) * 2pi
         local Q = RotZYZ(θ...)
         local D = transpose(WignerD.wignerD(Ltot, θ...)) 
         local QRs = [Q*Rs[i] for i in 1:length(Rs)]

         fRs1 = eval_sym_basis(Rs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         fRs1Q = eval_sym_basis(QRs; coeffs = C_rpe_recursive, MM = MM, ll = ll, nn = nn)
         Ltot == 0 ? (print_tf(@test norm(fRs1 - fRs1Q) < 1e-12)) : (print_tf(@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-12))


         # @info("Test that the two recursive RPEs span the same space as RPE")
         # Do the rand batch on the same set of points
         ntest = 1000
         ORD = length(ll) # length of each group 
         BB1 = complex.(zeros(typeof(C_rpe_recursive[1]), size(C_rpe_recursive, 1), ntest))
         BB2 = complex.(zeros(typeof(C_rpe[1]), size(C_rpe, 1), ntest))
         for i = 1:ntest 
            # construct a random set of particles with 𝐫 ∈ ball(radius=1)
            Rs = [ rand_ball() for _ in 1:ORD ]
            BB1[:, i] = eval_sym_basis(Rs; coeffs=C_rpe_recursive, MM=MM, ll=ll, nn=nn)
            BB2[:, i] = eval_sym_basis(Rs; coeffs=C_rpe, MM=MM, ll=ll, nn=nn) 
         end
         print_tf(@test rank(gram(BB1); rtol=1e-11) == rank(gram(BB2); rtol=1e-11) == rank(gram([BB1;BB2]); rtol=1e-11) == size(C_rpe,1))
      end
      println()
   end
end