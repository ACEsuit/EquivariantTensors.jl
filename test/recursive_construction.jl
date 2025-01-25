using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD,
      Combinatorics, Rotations
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real
O3 = RepLieGroups.O3
using RepLieGroups: gram, m_generate
using Test
using PartialWaveFunctions

# for i = 1:200
# ll = SA[rand(0:2, 5)...]
# nn = SA[rand(0:2, 5)...]
nn = SA[1,1,1,2,2,2]
ll = SA[1,2,3,1,2,3]
N1 = 3 # rand(1:length(ll)-1) # random partition
N2 = length(ll) - N1 # length of the second partition
ll1 = SA[ll[1:N1]...]
ll2 = SA[ll[N1+1:end]...]
nn1 = SA[nn[1:N1]...]
nn2 = SA[nn[N1+1:end]...]

Ltot = rand(0:4)
# if isodd(sum(ll)+Ltot); continue; end
if isodd(sum(ll)+Ltot); Ltot += 1; end
total_rank = 0
m_class = m_generate(nn,ll,Ltot)[1]
MM = []
for i = 1:length(m_class)
   for j = 1:length(m_class[i])
      push!(MM, m_class[i][j])
   end
end
MM = identity.(MM)
MM_dict = Dict(MM[i] => i for i = 1:length(MM))
T = Ltot == 0 ? Float64 : SVector{2Ltot+1, Float64}
C_re_semi_pi = []
t1 = @elapsed for L1 in 0:sum(ll1)
   for L2 in abs(L1-Ltot):minimum([L1+Ltot,sum(ll2)])
      # global C1, _,_, M1 = re_rpe(nn1,ll1,L1)
      # global C2, _,_, M2 = re_rpe(nn2,ll2,L2)
      global C1,M1 = rpe_basis_new(nn1,ll1,L1)
      global C2,M2 = rpe_basis_new(nn2,ll2,L2)
      # global Basis_func = Ltot == 0 ? zeros(Float64, size(C1,1),size(C2,1),length(M1)*length(M2)) : zeros(SVector{2Ltot+1, Float64}, size(C1,1),size(C2,1),length(M1)*length(M2))
      global counter = 0
      for i1 in 1:size(C1,1)
         for i2 in 1:size(C2,1)
            cc = [ zero(T) for _ = 1:length(MM) ]
            for (k1,m1) in enumerate(M1)
               for (k2,m2) in enumerate(M2)
                  if abs(sum(m1)+sum(m2))<=Ltot
                     k = MM_dict[SA[m1...,m2...]] # findfirst(m -> m == SA[m1...,m2...], MM)
                     # @assert !isnothing(k)
                     #    Basis_func[i1,i2,counter] = Ltot == 0 ? 
                     #       clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1] :
                     #       clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1]*I(2Ltot+1)[sum(m1)+sum(m2)+Ltot+1,:]
                     cc[k] = Ltot == 0 ? clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1] :
                                         clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1]*I(2Ltot+1)[sum(m1)+sum(m2)+Ltot+1,:] 
                  end
               end
            end
            push!(C_re_semi_pi, cc)
            counter += 1
         end
      end
      # BB = reshape(Basis_func,size(C1,1)*size(C2,1),length(M1)*length(M2))
      # BB = BB[:,1:length(MM)]
      # @assert length(MM) == counter-1

      # total_rank += rank(gram(BB))
      # @show size(BB), rank(gram(BB))
   end
end
C_re_semi_pi_final = identity.([C_re_semi_pi[i][j] for i = 1:length(C_re_semi_pi), j = 1:length(MM)])
t2 = @elapsed C_re,C_rpe,_,M = re_rpe(nn,ll,Ltot)
@show t1, t2
if rank(gram(C_rpe)) > 0
   @show rank(gram(C_re_semi_pi_final))
   @test rank(gram(C_rpe)) == rank(gram(C_re_semi_pi_final)) <= rank(gram(C_re))
end
# end