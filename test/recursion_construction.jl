using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD,
      Combinatorics, Rotations
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real
O3 = RepLieGroups.O3
using RepLieGroups: gram
using Test
using PartialWaveFunctions

ll = SA[rand(0:2, 6)...]
nn = SA[rand(0:2, 6)...]
N1 = rand(1:length(ll)-1) # random partition
N2 = length(ll) - N1 # length of the second partition
ll1 = SA[ll[1:N1]...]
ll2 = SA[ll[N1+1:end]...]
nn1 = SA[nn[1:N1]...]
nn2 = SA[nn[N1+1:end]...]

Ltot = rand(0:4)
total_rank = 0
t1 = @elapsed for L1 in 0:sum(ll1)
   for L2 in abs(L1-Ltot):L1+Ltot
      if L2 > sum(ll2); continue; end
      # global C1, _,_, M1 = re_rpe(nn1,ll1,L1)
      # global C2, _,_, M2 = re_rpe(nn2,ll2,L2)
      global C1,M1 = rpe_basis_new(nn1,ll1,L1)
      global C2,M2 = rpe_basis_new(nn2,ll2,L2)
      global Basis_func = Ltot == 0 ? zeros(Float64, size(C1,1),size(C2,1),length(M1)*length(M2)) : zeros(SVector{2Ltot+1, Float64}, size(C1,1),size(C2,1),length(M1)*length(M2))
      global counter = 1
      global MM = []
      for (k1,m1) in enumerate(M1)
         for (k2,m2) in enumerate(M2)
            if abs(sum(m1)+sum(m2))<=Ltot
               for i1 in 1:size(C1,1)
                  for i2 in 1:size(C2,1)
                     Basis_func[i1,i2,counter] = Ltot == 0 ? 
                        clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1] :
                        clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1]*I(2Ltot+1)[sum(m1)+sum(m2)+Ltot+1,:]
                  end
               end
            end
            counter += 1
            push!(MM, [m1;m2])
         end
      end
      BB = reshape(Basis_func,size(C1,1)*size(C2,1),length(M1)*length(M2))
      BB = BB[:,1:length(MM)]
      @assert length(MM) == counter-1

      total_rank += rank(gram(BB))
      @show size(BB), rank(gram(BB))
   end
end

t2 = @elapsed C_re,C_rpe,_,M = re_rpe(nn,ll,Ltot)
@show t1, t2
@test rank(gram(C_rpe)) <= total_rank <= rank(gram(C_re))
