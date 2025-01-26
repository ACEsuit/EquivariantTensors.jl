using StaticArrays, LinearAlgebra, RepLieGroups, WignerD, 
      Rotations
using Test

for ntest = 1:200
   ll = SA[rand(0:1, 6)...]
   nn = SA[rand(1:2, 6)...]
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

      # @info("Test that re_semi_pi span the same space as RPE")
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

llset = [SA[1,1,1,1], SA[1,1,2,2], SA[1,2,2,2], SA[1,1,2,2,2], SA[1,1,2,2,3], SA[1,1,1,2,2,3] ]
nnset = [SA[1,1,2,2], SA[1,1,2,3], SA[1,3,1,2], SA[1,2,1,2,2], SA[1,1,1,1,1], SA[2,2,1,1,2,1] ]
Partition = [2,2,1,2,2,3]


for k = 1:length(llset)
   nn = nnset[k]
   ll = llset[k]
   N1 = Partition[k]
   
   for ntest = 1:20
      Ltot = iseven(sum(ll)) ? rand(0:2:4) : rand(1:2:3)
      t1 = @elapsed C_re_semi_pi, MM = re_semi_pi(nn,ll,Ltot,N1)
      t2 = @elapsed C_rpe,M = rpe_basis_new(nn,ll,Ltot)
      @show t1, t2
   
      if rank(gram(C_rpe)) > 0
         # @info("Test that re_semi_pi span a set with rank ranging between RE and RPE")
         @test rank(gram(C_rpe)) == rank(gram(C_re_semi_pi))

         # @info("Testing the equivariance of the old RPE basis")
         Rs = rand_config(length(ll))
         θ = rand(3) * 2pi
         Q = RotZYZ(θ...)
         D = transpose(WignerD.wignerD(Ltot, θ...)) 
         QRs = [Q*Rs[i] for i in 1:length(Rs)]
         fRs1 = eval_basis(Rs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
         fRs1Q = eval_basis(QRs; coeffs = C_re_semi_pi, MM = MM, ll = ll, nn = nn)
         Ltot == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14)

         # @info("Test that re_semi_pi span the same space as RPE")
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
         @test rank(gram(C_re_semi_pi)) == rank(gram(BB1); rtol=1e-11) == rank(gram([BB1;BB2]); rtol=1e-11) == rank(gram(BB2); rtol=1e-11) == rank(gram(C_rpe))
      end
   end
end