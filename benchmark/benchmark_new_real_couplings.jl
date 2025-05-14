using Test, EquivariantTensors, StaticArrays, SpheriCart, Combinatorics
# using EquivariantTensors.O3: Ctran, coupling_coeffs, gram
# using EquivariantTensors.O3new: Ctran, coupling_coeffs, gram
using PartialWaveFunctions: clebschgordan
using LinearAlgebra, Random
using WignerD, Rotations, BlockDiagonals
using BenchmarkTools

import EquivariantTensors.O3 as O3

isdefined(Main, :___UTILS_FOR_TESTS___) || include("../test/utils/utils_testO3.jl")

# The full test set is quite large and takes a while, for a quick sanity test 
# it is better to keep the test small so it runs quickly. 
#
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

@info("Equivariance and Linear Independence of coupled cSH & rSH based basis")

for L = 0:Lmax 
   global Lmax, lmax, nmax, ORDmax
   local θ, ll, Ure, Ure_r, Urpe, Urpe_r, Mll, Mll_r 
   local Ure_new, Ure_r_new, Urpe_new, Urpe_r_new, Mll_new, Mll_r_new 
   local X, Q, B1, B2, B3, B4, B5, B6, B7, B8
   local rk, rk_r, ntest
   local BB, BB_r, BB_sym, BB_sym_r

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

   verbose = true 

   @info("Using ultra short nnll list for testing the case L = $L")
   nnll_list = ultra_short_nnll_list

   for (itest, (nn, ll)) in enumerate(nnll_list)
      nn = shuffle(nn)
      ll = shuffle(ll)
      N = length(ll)
      @assert length(ll) == length(nn)
      @show nn, ll

      @btime Ure, Mll = O3.coupling_coeffs($L, $ll, $nn; PI = false) # cSH based re_basis
      Ure, Mll = O3.coupling_coeffs(L, ll, nn; PI = false) # cSH based re_basis
      @btime Ure_r, Mll_r = O3.coupling_coeffs($L, $ll, $nn; PI = false, basis = real) # rSH based re_basis
      Ure_r, Mll_r = O3.coupling_coeffs(L, ll, nn; PI = false, basis = real) # rSH based re_basis
      @btime Urpe, Mll_rpe = O3.coupling_coeffs($L, $ll, $nn) # cSH based rpe_basis
      Urpe, Mll_rpe = O3.coupling_coeffs(L, ll, nn) # cSH based rpe_basis
      @btime Urpe_r, Mll_r_rpe = O3.coupling_coeffs($L, $ll, $nn; basis = real) # rSH based rpe_basis
      Urpe_r, Mll_r_rpe = O3.coupling_coeffs(L, ll, nn; basis = real) # rSH based rpe_basis
   end
   println()
end
