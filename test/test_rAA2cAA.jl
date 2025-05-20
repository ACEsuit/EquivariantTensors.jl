using Test, EquivariantTensors, StaticArrays
using EquivariantTensors.O3: mm_generate, rAA2cAA
using LinearAlgebra

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

___O3_TESTS___ = :small 
# :___O3_TESTS___ = :large


##

if ___O3_TESTS___ == :small
   @info("Using SMALL couplings test set")
   Lmax = 2
   lmax = 4 
   nmax = 4 
   ORDmax = 3 
elseif ___O3_TESTS___ == :large
   @info("Using LARGE couplings test set")
   Lmax = 4
   lmax = 4 
   nmax = 4 
   ORDmax = 5 
end

@info("Testing the correctness of rAA2cAA")

for L = 0:Lmax 
    local X
 
    # generate an nnll list for each L for testing
    nnll_list = [] 
 
    for ORD = 2:ORDmax
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
 
    @info("Using ultra short nnll list for testing the case L = $L")
    nnll_list = ultra_short_nnll_list

    for (nn, ll) in nnll_list

        # Generate random points on the sphere
        X = [ (0.1 + 0.9 * rand()) * rand_sphere() for i in 1:length(ll) ]
        MM_c = mm_generate(L, ll, nn)
        MM_r = mm_generate(L, ll, nn; flag = :SpheriCart)
        AA_c = eval_AA_basis(X; MM = MM_c, ll, nn, sym = false)
        AA_r = eval_AA_basis(X; MM = MM_r, ll, nn, Real = true, sym = false)

        C = rAA2cAA(MM_c, MM_r)

        print_tf(@test norm(C * AA_r - AA_c) < 1e-12)
    end
    println()
end