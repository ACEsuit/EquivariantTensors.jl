

using EquivariantTensors.O3: yvector, SYYVector, _lm2i, _i2lm
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

## 

@info("Some basic alternative constructor tests") 
y0 = rand(1); y1 = rand(3); y2 = rand(5) 
y012 = vcat(y0, y1, y2)
println_slim(@test yvector(y0, y1, y2) == SYYVector(tuple(y012...)))

y1[:] .= 0 
println_slim(@test yvector(y0, y1, y2) == yvector(y0, nothing, y2))

y0 = 0.0
println_slim(@test yvector(y0, y1, y2) == yvector(y0, nothing, y2))

function check_alc(z0, z1, z2) 
   yvector(z0, z1, z2)
   return @allocated yvector(z0, z1, z2)
end

z0 = rand(); z1 = nothing; z2 = @SVector rand(5)
# @code_warntype yvector(z0, z1, z2)
println_slim(@test check_alc(z0, z1, z2) == 0) 

z0 = @SVector rand(1); z1 = @SVector zeros(3); z2 = @SVector rand(5)
# @code_warntype yvector(z0, z1, z2)
println_slim(@test check_alc(z0, z1, z2) == 0) 
