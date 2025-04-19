
using StaticArrays, LinearAlgebra, EquivariantTensors, WignerD, Combinatorics, 
      Rotations
using WignerD: wignerD
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("../utils/utils_for_tests.jl")

##

# Test the new RPE basis up to L = 4
@info("Testing the new cSH-based RPE basis")
for L = 0:4
   @info("Testing L = $L")

   # generate a nnll_list for testing
   lmax = 4
   nmax = 4
   nnll_list = [] 

   for ORD = 2:6
      for ll in with_replacement_combinations(1:lmax, ORD) 
         # 0 or 1 above ?
         if !iseven(sum(ll)+L); continue; end  # This is to ensure the reflection symmetry
         if sum(ll) > 2 * lmax; continue; end 
         for Inn in CartesianIndices( ntuple(_->1:nmax, ORD) )
            nn = [ Inn.I[α] for α = 1:ORD ]
            if sum(nn) > sum(1:nmax); continue; end
            nnll = [ (ll[α], nn[α]) for α = 1:ORD ]
            if !issorted(nnll); continue; end
            push!(nnll_list, (SVector(nn...), SVector(ll...)))
         end
      end
   end

   long_nnll_list = nnll_list 
   short_nnll_list = nnll_list[1:10:end]
   ultra_short_nnll_list = nnll_list[1:100:end]

   verbose = true 

   @info("Using short nnll list for testing")
   nnll_list = short_nnll_list

   for (itest, (nn, ll)) in enumerate(nnll_list)
      N = length(ll)
      @assert length(ll) == length(nn)
      
      local Rs, θ, Q, QRs, D, D_r

      Rs = rand_config(length(ll))
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      D = transpose(wignerD(L, θ...)) 
      D_r = Ctran(L) * D * Ctran(L)'
      QRs = [Q*Rs[i] for i in 1:length(Rs)]
      
      t_rpe = @elapsed coeffs_ind, MM = O3.rpe_basis_new(nn,ll,L)
      t_rpe_r = @elapsed coeffs_ind_r, MM_r = O3.rpe_basis_new(nn,ll,L; flag = :SpheriCart)
      rk = rank(O3.gram(coeffs_ind),rtol = 1e-12)
      rk_r = rank(O3.gram(coeffs_ind_r),rtol = 1e-12)
      @test rk == rk_r

      fRs1 = eval_basis(Rs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn)
      fRs1Q = eval_basis(QRs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn)
      # @info("Testing the equivariance of the old RPE basis")
      L == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14)

      fRs1_r = eval_basis(Rs; coeffs = coeffs_ind_r, MM = MM_r, ll = ll, nn = nn, Real = true)
      fRs1Q_r = eval_basis(QRs; coeffs = coeffs_ind_r, MM = MM_r, ll = ll, nn = nn, Real = true)
      # @info("Testing the equivariance of the old RPE basis")
      L == 0 ? (@test norm(fRs1_r - fRs1Q_r) < 1e-14) : (@test norm(fRs1_r - Ref(D_r) .* fRs1Q_r) < 1e-14)

      ntest = 1000

      RR = make_batch(ntest, length(ll))

      X = rand_batch(; coeffs=coeffs_ind, MM=MM, ll=ll, nn=nn, batch = RR)
      @test rank(O3.gram(X); rtol=1e-12) == size(X,1) == rk

      X_r = rand_batch(; coeffs=coeffs_ind_r, MM=MM_r, ll=ll, nn=nn, batch = RR)
      @test rank(O3.gram(X_r); rtol=1e-12) == size(X_r,1) == rk_r

      Xsym = sym_rand_batch(; coeffs=coeffs_ind, MM=MM, ll=ll, nn=nn, batch = RR)
      @test rank(O3.gram(Xsym); rtol=1e-12) == size(Xsym,1) == rk

      Xsym_r = sym_rand_batch(; coeffs=coeffs_ind_r, MM=MM_r, ll=ll, nn=nn, batch = RR)
      @test rank(O3.gram(Xsym_r); rtol=1e-12) == size(Xsym_r,1) == rk_r

      if verbose 
         @info("Test $itest: t_rpe = $t_rpe, t_rpe_r = $t_rpe_r")
      else 
         print(".")
      end
   end
end