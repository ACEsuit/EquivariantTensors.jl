using StaticArrays
using EquivariantTensors.O3: coupling_coeffs

# The efficiency of the new recursive method is shown here, 
# which can also be an example showing that how much we may get if we 
# store in advance some coupling coefficients.
# The advantages, however, only become significant when the correlation 
# order is high. 

for N = 6:10
   nn = SA[ones(Int64,N)...] .* rand(1:5)
   ll = SA[ones(Int64,N)...]
   N1 = Int(round(N/2))

   for Ltot in (iseven(sum(ll)) ? (0:2:4) : (1:2:3))
      t_rpe = @elapsed C_rpe, M = coupling_coeffs(Ltot,ll,nn) # reference time
      t_rpe_recursive_kernel = @elapsed C_rpe_recursive, MM = coupling_coeffs(Ltot,ll,nn,N1; symmetrization_method = :kernel) 
      # time for rpe_basis_new with kernel symmetrization - the time for explicit symmetrization is not shown here because it is less efficient

      println("Case : nn = $nn, ll = $ll, Ltot = $Ltot, N1 = $N1")
      println("Standard RPE basis : $t_rpe")
      println("Recursive RPE basis : $t_rpe_recursive_kernel")
      println()

    # below are correctness check, which has already been done in the tests
    #   if size(C_rpe_recursive,1) == size(C_rpe,1) != 0
    #      if MM != M
    #         @assert sort(MM) == sort(M)
    #         ord = sortperm(MM)
    #         @assert MM[ord] = sort(MM)
    #         C_rpe_recursive = C_rpe_recursive[:,ord]
    #         ord = sortperm(M)
    #         @assert M[ord] = sort(M)
    #         C_rpe = C_rpe[:,ord]
    #         MM = sort(MM)
    #      end
    #   end

    #   @test size(C_rpe,1) == size(C_rpe_recursive,1) == rank(gram([C_rpe;C_rpe_recursive]), rtol=1e-11)
   end
end