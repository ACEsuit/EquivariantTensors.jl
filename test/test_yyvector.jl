

using EquivariantTensors: SYYVector, _lm2i, _i2lm
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_testO3.jl")

##

for L = 0:4
   local data, y
   println()
   @info("Tests for L = $L...")
   println()
   data = tuple(randn((L+1)^2)...)
   y = SYYVector(data);
   
   @info("Test whether y[i] is as expected.")
   for i = 1 : (L+1)^2
      print_tf(@test y[i] == data[i])
   end
   println()
   
   @info("test whether y[(l,m)] is as expected.")
   for l = 0:L, m = -l:l
      print_tf(@test y[(l,m)] == y[(l=l,m=m)] == data[_lm2i(l,m)])
   end 
   for i = 1 : (L+1)^2
      print_tf(@test y[_i2lm(i)] == data[i]) 
   end
   println()
   
   @info("test whether y[Val(l)] is as expected.")
   for l = 0:L
      print_tf(@test y[Val(l)] == [data...][l^2+1:(l+1)^2])
      # Redundant but can serve as an "cross validation"...
      print_tf(@test y[Val(l)] == y[l^2+1:(l+1)^2]) 
   end
   println()
end
